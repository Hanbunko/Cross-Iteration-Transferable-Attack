import torch
import random
import numpy as np
import copy
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
import os

# Set random seeds for reproducibility
see = 314159265
random.seed(see)
np.random.seed(see)
torch.manual_seed(see)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(see)

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {dev}")

# Create directory for saving models
os.makedirs('./flm_bert', exist_ok=True)

# Load and preprocess AG News dataset
print("Loading tokenizer and preparing data...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

# Load dataset
print("Loading AG News dataset...")
dataset = load_dataset('ag_news')
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

# Create data loaders
tra_set = tokenized_datasets['train']
tes_set = tokenized_datasets['test']

# Create test loader
test_loader = DataLoader(tes_set, batch_size=32, shuffle=False)

# Federated setting
num_cli = 1000
indices = list(range(len(tra_set)))
random.shuffle(indices)
loc_siz = len(indices) // num_cli
lotl = []  # List of client data loaders
for idx in range(0, len(tra_set), loc_siz):
    lotl.append(DataLoader(
        Subset(tra_set, indices[idx:idx+loc_siz]),
        shuffle=True,
        batch_size=16
    ))

# Initialize global model
print("Initializing BERT model...")
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=4
).to(dev)

# Offset for continuing training from a checkpoint
ofs = 0
# Uncomment to load from checkpoint:
# model.load_state_dict(torch.load(f"./flm_bert/{ofs}.pth"))
learnrate = 2e-5
# Global optimizer (for potential server-side optimization if needed)
optimizer = AdamW(model.parameters(), lr=learnrate, weight_decay=0.01)

def train(model, train_loader, epochs=1):
    """Local training function for each client"""
    model.train()
    # Create a fresh optimizer for local training
    local_optimizer = AdamW(model.parameters(), lr=learnrate, weight_decay=0.01)

    for epoch in range(epochs):
        for idx, batch in enumerate(train_loader):
            # Move batch to device
            input_ids = batch['input_ids'].to(dev)
            attention_mask = batch['attention_mask'].to(dev)
            labels = batch['labels'].to(dev)

            local_optimizer.zero_grad()

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            local_optimizer.step()

# Initialize global model state
glomod = copy.deepcopy(model.state_dict())

# Federated Learning parameters
sam_rat = 0.01
nor = 101  # Number of rounds

print("\nStarting Federated Learning...")
for r in range(nor):
    print(f"\n=== FL Round {r}/{nor} ===")

    # Sample clients for this round
    nop = max(1, int(num_cli * sam_rat))  # Number of participating clients
    par = random.sample(range(num_cli), nop)

    # Initialize accumulators for aggregating client updates
    accumulators = {key: torch.zeros_like(param, device=dev)
                   for key, param in glomod.items()}

    # Train on selected clients
    for i, p in enumerate(par):
        print(f"Training client {i+1}/{len(par)} (Client ID: {p})")

        # Load the global model for this client
        model.load_state_dict(copy.deepcopy(glomod))

        # Local training
        model.train()
        train(model, lotl[p], epochs=1)  # 1 epoch per client per round

        # Accumulate client updates
        for key, param in model.state_dict().items():
            accumulators[key] = accumulators[key] + copy.deepcopy(param.detach())

    # Federated Averaging: average all client updates
    for key, param in model.state_dict().items():
        if 'num_batches_tracked' in key:
            # Skip batch norm tracking statistics
            continue
        glomod[key] = accumulators[key] / len(par)

    # Evaluate global model
    model.eval()
    model.load_state_dict(glomod)

    correct_clean = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(dev)
            attention_mask = batch['attention_mask'].to(dev)
            labels = batch['labels'].to(dev)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct_clean += (predicted == labels).sum().item()

    accuracy = 100 * correct_clean / total
    print(f'FL Accuracy at round {r}: {accuracy:.2f}%')

    # Save checkpoint every 10 rounds
    if r % 1 == 0:
        torch.save(model.state_dict(), f'./bertmod2/{r+ofs}.pth')
        print(f"Saved checkpoint at round {r}")

print("\nFederated Learning completed!")