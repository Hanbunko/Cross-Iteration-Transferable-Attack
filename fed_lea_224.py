import torch
import random
import numpy as np
import copy
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torchvision.models as models
from torch.nn import functional as F

# Resize to 224*224 image size, we did not use this.

see = 314159265
random.seed(see)
np.random.seed(see)
torch.manual_seed(see)

dev = torch.device('cuda')

# CIFAR-10 dataset with 224x224 resizing
tra_tran, tes_tran = [], []
mea = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)
siz = 4

# Add resize to 224x224 for both training and testing
tra_tran += [transforms.Resize((224, 224))]
tra_tran += [transforms.RandomCrop(224, padding=siz), transforms.RandomHorizontalFlip()]
tra_tran += [transforms.ToTensor()]
tes_tran += [transforms.Resize((224, 224))]
tes_tran += [transforms.ToTensor()]

norm = transforms.Normalize(mea, std)
tra_tran += [norm]
tes_tran += [norm]

tra_set = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.Compose(tra_tran))
tes_set = datasets.CIFAR10('./data', train=False, download=True, transform=transforms.Compose(tes_tran))

train_loader = DataLoader(tra_set, batch_size=64, shuffle=True)
test_loader = DataLoader(tes_set, batch_size=64, shuffle=False)

# Federated setting
num_cli = 100
indices = list(range(len(tra_set)))
random.shuffle(indices)
loc_siz = len(indices) // num_cli
lotl = []
for idx in range(0, len(tra_set), loc_siz):
    lotl.append(DataLoader(Subset(tra_set, indices[idx:idx+loc_siz]), shuffle=True, batch_size=64))
ofs = 0

# Use torchvision's ResNet18 pretrained on ImageNet (designed for 224x224)
model = models.resnet18(pretrained=False, num_classes=10).to(dev)
# Alternatively, if you want to use pretrained weights and fine-tune:
# model = models.resnet18(pretrained=True)
# model.fc = nn.Linear(model.fc.in_features, 10)  # Replace final layer for 10 classes
# model = model.to(dev)

# model = torch.load(f"./flm/{ofs}.pth")
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
criterion = nn.CrossEntropyLoss()

def train(model, train_loader, epochs=5):
    model.train()
    for epoch in range(epochs):
        for idx, (data, target) in enumerate(train_loader):
            data, target = data.to(dev), target.to(dev)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {idx}/{len(train_loader)}, Loss: {loss.item()}")
        # scheduler.step()

glomod = copy.deepcopy(model.state_dict())
sam_rat = 0.1
nor = 1001

for r in range(nor):
    nop = int(num_cli * sam_rat)
    par = random.sample(range(num_cli), nop)
    accumulators = {key: torch.zeros_like(param, device=dev) for key, param in glomod.items()}

    for p in par:
        # Load the global model
        model.load_state_dict(copy.deepcopy(glomod))
        # Local train
        model.train()
        train(model, lotl[p])
        for key, param in model.state_dict().items():
            accumulators[key] = accumulators[key] + copy.deepcopy(param.detach())

    for key, param in model.state_dict().items():
        if 'num_batches_tracked' in key:
            continue
        glomod[key] = accumulators[key] / len(par)

    model.eval()
    correct_clean = 0
    total = 0
    model.load_state_dict(glomod)

    if r % 100 == 0:
        torch.save(model, f'./mod224/{r+ofs}.pth')

    for images, labels in test_loader:
        images, labels = images.to(dev), labels.to(dev)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct_clean += (predicted == labels).sum().item()

    print(f'FL Accuracy at round {r}: %.2f %%' % (100 * correct_clean / total))
