import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import copy
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

device = "cuda:0"

def get_embedding_layer(model):
    """Get the embedding layer from a BERT model"""
    if hasattr(model, 'bert'):
        return model.bert.embeddings.word_embeddings
    elif hasattr(model, 'roberta'):
        return model.roberta.embeddings.word_embeddings
    elif hasattr(model, 'distilbert'):
        return model.distilbert.embeddings.word_embeddings
    else:
        raise ValueError("Unsupported model type")


def pgd_targeted_embeddings(model, input_ids, attention_mask, target_labels,
                           epsilon=0.3, alpha=0.01, iters=20):
    """Targeted PGD attack on embedding space"""
    device = next(model.parameters()).device
    model.eval()

    emb_layer = get_embedding_layer(model)
    with torch.no_grad():
        original = emb_layer(input_ids.to(device))

    perturbed = original.clone().detach().requires_grad_(True)

    for _ in range(iters):
        model.zero_grad(set_to_none=True)
        outputs = model(inputs_embeds=perturbed, attention_mask=attention_mask.to(device))
        loss = F.cross_entropy(outputs.logits, target_labels.to(device))
        loss.backward()

        grad = perturbed.grad.detach()
        perturbed.data = perturbed.data - alpha * grad.sign()
        eta = torch.clamp(perturbed.data - original.data, -epsilon, epsilon)
        perturbed.data = original.data + eta
        perturbed.grad = None

    return perturbed.detach()


def create_diversified_ensemble_imdb(early_model, tokenizer, train_texts,
                                     num_models=10, max_length=256, batch_size=32,
                                     device='cuda', save_dir='./imdb_models'):
    """
    Create ensemble using IMDB texts with AGNews model
    Similar to using CIFAR-100 images with CIFAR-10 model

    Args:
        early_model: Initial BERT model trained on AGNews (4 classes)
        tokenizer: BERT tokenizer
        train_texts: List of IMDB training texts (labels are ignored, only texts used)
        num_models: Number of models to create in ensemble
        max_length: Maximum sequence length for tokenization
        batch_size: Batch size for processing
        device: Device to use ('cuda' or 'cpu')
        save_dir: Directory to save models

    Returns:
        List of diversified models
    """
    ensemble_models = [early_model]
    num_classes = 4  # AGNews has 4 classes - we target all 4 classes

    for tc in range(num_models):
        print(f"\nGenerating Model {tc+1}:")

        all_adv_embeds = []
        all_attention_masks = []
        all_soft_labels = []

        # Process in batches
        for i in range(0, len(train_texts), batch_size):
            batch_texts = train_texts[i:i+batch_size]

            # Tokenize batch
            encoded = tokenizer(
                batch_texts,
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='pt'
            )

            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)

            # Get soft labels (model outputs) from AGNews model on IMDB texts
            with torch.no_grad():
                outputs = ensemble_models[-1](input_ids=input_ids, attention_mask=attention_mask)
                soft_labels = outputs.logits  # Keep logits as soft labels

            # Get predicted AGNews classes
            _, predicted = soft_labels.max(1)

            # Generate random target labels for AGNews classes (0-3)
            # Just like your CIFAR code uses random CIFAR-10 classes
            new_labels = torch.randint(0, num_classes, predicted.size(), device=device)

            # Generate adversarial embeddings targeting random AGNews classes
            adv_embeds = pgd_targeted_embeddings(
                ensemble_models[-1],
                input_ids,
                attention_mask,
                new_labels,
                epsilon=0.1,
                alpha=0.01,
                iters=10
            )

            all_adv_embeds.append(adv_embeds.cpu())
            all_attention_masks.append(attention_mask.cpu())
            all_soft_labels.append(soft_labels.cpu())

        # Create dataset with adversarial embeddings and ORIGINAL soft labels
        adv_embeds_tensor = torch.cat(all_adv_embeds)
        attention_masks_tensor = torch.cat(all_attention_masks)
        soft_labels_tensor = torch.cat(all_soft_labels)

        adv_dataset = TensorDataset(adv_embeds_tensor, attention_masks_tensor, soft_labels_tensor)
        adv_loader = DataLoader(adv_dataset, batch_size=batch_size, shuffle=True)

        # Train new model on adversarial examples with soft labels
        model = copy.deepcopy(early_model).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        # Use MSE loss for soft label training
        mse_criterion = nn.MSELoss()

        adv_epochs = 10
        for epoch in range(adv_epochs):
            model.train()
            total_loss = 0

            for inputs_embeds, attention_mask, soft_targets in adv_loader:
                inputs_embeds = inputs_embeds.to(device)
                attention_mask = attention_mask.to(device)
                soft_targets = soft_targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
                loss = mse_criterion(outputs.logits, soft_targets)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(adv_loader)
            print(f"  Epoch {epoch+1}/{adv_epochs} | MSE Loss: {avg_loss:.4f}")

        model.eval()
        # Save model
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f'model_{tc+1}.pth')
        torch.save(model.state_dict(), model_path)
        print(f"  Model saved to {model_path}")

        ensemble_models.append(model)

    return ensemble_models


def evaluate_model_on_agnews(model, tokenizer, test_texts, test_labels,
                             max_length=128, batch_size=32, device='cuda'):
    """Evaluate a single model on AGNews test set"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for i in range(0, len(test_texts), batch_size):
            batch_texts = test_texts[i:i+batch_size]
            batch_labels = test_labels[i:i+batch_size]

            # Tokenize
            encoded = tokenizer(
                batch_texts,
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='pt'
            )

            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            labels_tensor = torch.tensor(batch_labels, dtype=torch.long).to(device)

            # Get predictions
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = outputs.logits.argmax(dim=1)

            correct += (predictions == labels_tensor).sum().item()
            total += len(batch_labels)

    accuracy = 100.0 * correct / total
    return accuracy


# Example usage
def main():
    # Load IMDB dataset - we only use the texts, not the labels
    imdb_dataset = load_dataset('imdb')
    imdb_train_texts = imdb_dataset['train']['text']

    # Load AGNews dataset for evaluation
    agnews_dataset = load_dataset('ag_news')
    agnews_test_texts = agnews_dataset['test']['text'][:1000]  # Use subset
    agnews_test_labels = agnews_dataset['test']['label'][:1000]

    # Load pre-trained BERT model for AGNews (4 classes)
    # This should be a model already trained on AGNews
    model_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Initialize or load a model trained on AGNews (4 classes)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=4  # AGNews has 4 classes
    )
    # In practice, you would load a checkpoint that's already trained on AGNews:
    # model.load_state_dict(torch.load('agnews_trained_model.pth'))
    model.load_state_dict(torch.load("bertmod/1.pth"))
    model.eval()
    model = model.to(device)

    # Create diversified ensemble
    ensemble = create_diversified_ensemble_imdb(
        early_model=model,
        tokenizer=tokenizer,
        train_texts=imdb_train_texts,
        num_models=10,  # Create 5 additional models
        max_length=256,
        batch_size=16,
        device=device,
        save_dir='./text_ens_1'
    )

    print(f"\nEnsemble created with {len(ensemble)} models")

    # Test all models on AGNews and calculate average accuracy
    print("\n" + "="*50)
    print("Evaluating ensemble on AGNews test set:")
    print("="*50)

    accuracies = []
    for idx, model in enumerate(ensemble):
        accuracy = evaluate_model_on_agnews(
            model, tokenizer,
            agnews_test_texts, agnews_test_labels,
            max_length=128,  # AGNews uses shorter sequences
            batch_size=32,
            device=device
        )
        accuracies.append(accuracy)
        print(f"Model {idx}: {accuracy:.2f}%")

    avg_accuracy = sum(accuracies) / len(accuracies)
    print(f"\nAverage accuracy across all {len(ensemble)} models: {avg_accuracy:.2f}%")

    return ensemble, accuracies


if __name__ == "__main__":
    ensemble = main()