import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.fft
import tqdm
import copy
import os
import random
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets
import torchvision
import cv2
from PIL import Image
import torchattacks

import seaborn as sns
from sklearn.metrics import confusion_matrix

# Randomly initialize each new model

src_dic = "modsp"
device = "cuda:0"

leak_round = 100

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        out1 = F.relu(self.bn1(self.conv1(x)))
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        out6 = F.avg_pool2d(out5, 4)
        out7 = out6.view(out6.size(0), -1)
        out8 = self.linear(out7)

        if return_features:
            return out1, out2, out3, out4, out5, out6, out7, out8
        return out8

def ResNet18(num_classes=10):
    """ResNet18 for CIFAR-10 (but will use CIFAR-100 images)"""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

def prepare_cifar10(num_clients = 1, batch_size=64, device='cpu'):
    # Model
    model = ResNet18(num_classes=10).to(device).eval()

    # Dataset loaders
    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=test_transform)

    indices = list(range(len(train_dataset)))
    random.shuffle(indices)  # IID
    local_dataset_size = len(indices) // num_clients
    train_loaders = []
    for offset in range(0, len(train_dataset), local_dataset_size):
        train_loaders.append(DataLoader(Subset(train_dataset, indices[offset:offset + local_dataset_size]),
                                        batch_size=batch_size, shuffle=True))
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return model, train_loaders, test_loader, train_dataset.classes

def prepare_cifar100(num_clients=1, batch_size=64, device='cpu'):
    """Prepare CIFAR-100 dataset but with CIFAR-10 model (10 classes)"""
    # Model for CIFAR-10 (10 classes) but will process CIFAR-100 images
    model = ResNet18(num_classes=10).to(device).eval()

    # Dataset loaders for CIFAR-100
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    test_transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.CIFAR100('./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR100('./data', train=False, download=True, transform=test_transform)

    indices = list(range(len(train_dataset)))
    random.shuffle(indices)  # IID
    local_dataset_size = len(indices) // num_clients
    train_loaders = []
    for offset in range(0, len(train_dataset), local_dataset_size):
        train_loaders.append(DataLoader(
            Subset(train_dataset, indices[offset:offset + local_dataset_size]),
            batch_size=batch_size, shuffle=True
        ))
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return model, train_loaders, test_loader

def create_diversified_ensemble_cifar100(early_model, train_loader, num_models=10):
    """Create ensemble using CIFAR-100 images with CIFAR-10 model"""
    ensemble_models = [early_model]

    for tc in range(num_models):
        print(f"\nGenerating Model {tc+1}:")

        all_adv_images = []
        all_soft_labels = []

        # Targeted PGD attack
        atk = torchattacks.PGD(ensemble_models[-1], eps=8/255, alpha=2/255, steps=20)
        atk.set_mode_targeted_by_label()

        for batch_idx, (images, _) in tqdm.tqdm(enumerate(train_loader),
                                                 total=len(train_loader),
                                                 desc="Generating Adv Training Set"):
            images = images.to(device)

            # Get soft labels (model outputs) from CIFAR-10 model on CIFAR-100 images
            with torch.no_grad():
                temperature = 100
                logits = ensemble_models[0](images)
                soft_labels = logits
                # soft_labels = nn.functional.softmax(logits / temperature, dim=1)

            for target_label in range(10):
                new_labels = torch.full_like(_, target_label, device=device)

                images.requires_grad = True
                adv_images = atk(images, new_labels)

                all_adv_images.append(adv_images.cpu())
                all_soft_labels.append(soft_labels.cpu())

        # Create dataset with adversarial images and ORIGINAL soft labels
        adv_images_tensor = torch.cat(all_adv_images)
        soft_labels_tensor = torch.cat(all_soft_labels)

        adv_dataset = TensorDataset(adv_images_tensor, soft_labels_tensor)
        adv_loader = DataLoader(adv_dataset, batch_size=128, shuffle=True)

        # Train new model on adversarial examples with soft labels
        # Random initialize
        model = ResNet18(num_classes=10).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        def distillation_loss(student_logits, teacher_logits, temperature=6.0):
            """Knowledge distillation loss using KL divergence"""
            student_soft = F.log_softmax(student_logits / temperature, dim=1)
            teacher_soft = F.softmax(teacher_logits / temperature, dim=1)
            kl_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
            return kl_loss * (temperature ** 2)

        distillation_temperature = 100

        adv_epochs = 10
        for epoch in range(adv_epochs):
            model.train()
            total_loss = 0

            for inputs, soft_targets in adv_loader:
                inputs = inputs.to(device)
                soft_targets = soft_targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = distillation_loss(outputs, soft_targets, temperature=distillation_temperature)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(adv_loader)
            print(f"  Epoch {epoch+1}/{adv_epochs} | Loss: {avg_loss:.6f}")

        model.eval()
        # Save model
        os.makedirs(f'./nee_00_models_{leak_round}', exist_ok=True)
        torch.save(model, f'./nee_00_models_{leak_round}/{tc+1}.pth')

        ensemble_models.append(model)

    return ensemble_models

# ===============================
# Enhanced Ensemble MIFGSM Attack
# ===============================
class EnsembleMIFGSM:
    """Ensemble MI-FGSM for CIFAR-100"""

    def __init__(self, models, eps=8/255, alpha=2/255, steps=10, decay=1.0, device='cuda:0'):
        self.models = models
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.decay = decay
        self.device = device

    def __call__(self, images, labels, targeted=False):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        momentum = torch.zeros_like(images).detach().to(self.device)
        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True

            # Accumulate gradients from all models
            ensemble_grad = torch.zeros_like(images).to(self.device)

            for model in self.models:
                outputs = model(adv_images)

                if targeted:
                    cost = -loss(outputs, labels)
                else:
                    cost = loss(outputs, labels)

                grad = torch.autograd.grad(
                    cost, adv_images, retain_graph=False, create_graph=False
                )[0]

                ensemble_grad += grad

            # Average gradient
            ensemble_grad = ensemble_grad / len(self.models)

            # Normalize gradient
            ensemble_grad = ensemble_grad / torch.mean(
                torch.abs(ensemble_grad), dim=(1, 2, 3), keepdim=True
            )

            # Apply momentum
            ensemble_grad = ensemble_grad + momentum * self.decay
            momentum = ensemble_grad

            # Update adversarial images
            adv_images = adv_images.detach() + self.alpha * momentum.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images

def eval_model(model, test_loader):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels.to(device)).sum().item()

    return 100. * correct / total

def imshow(img, pth="default"):
    """Save adversarial example visualization"""
    npimg = img.cpu().detach().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(f'{pth}.png')
    plt.show()

# ===============
# Main Function
# ===============
def main():
    # exp_00: previous one
    targeted_class = -1  # -1 for untargeted attack
    num_ensemble_models = 40  # Number of models in ensemble

    set_seed(2025)

    _, trains, _ = prepare_cifar100(batch_size=128, device=device)
    _, _, test_loader, classes = prepare_cifar10(batch_size=128, device=device)

    final_model = torch.load(f'{src_dic}/1000.pth')
    final_model.to(device)
    final_model.eval()

    early_model = torch.load(f'{src_dic}/{leak_round}.pth')
    early_model.to(device)
    early_model.eval()

    '''
    print(f"ACC of original: {eval_model(early_model, test_loader)}")
    model = torch.load(f'nee_00_models_100/1.pth')
    print(f"ACC of new new: {eval_model(model, test_loader)}")
    return 0
    '''

    # Create diversified ensemble using soft labels
    print(f"\nCreating ensemble of {num_ensemble_models} models...")
    ensemble_models = create_diversified_ensemble_cifar100(
        early_model=early_model,
        train_loader=trains[0],
        num_models=num_ensemble_models
    )

    loacc = []
    oriacc = eval_model(early_model, test_loader)
    for i, mod in enumerate(ensemble_models):
        acc = eval_model(mod, test_loader)
        loacc.append(acc)

    print(f"\nEnsemble Average Accuracy: {sum(loacc)/len(loacc):.2f}%")
    print(f"Original Model Accuracy: {oriacc:.2f}%")

    # Initialize attacks
    ensemble_attack = EnsembleMIFGSM(
        models=ensemble_models,
        eps=8/255,
        alpha=2/255,
        steps=20,
        decay=1.0,
        device=device
    )

    single_attack = torchattacks.MIFGSM(early_model, eps=8/255, alpha=2/255, steps=20)

    # Evaluate attacks
    print("\n" + "="*60)
    print("Generating and evaluating adversarial examples...")
    print("="*60)

    total = 0
    ensemble_succ = 0
    single_succ = 0

    for batch_idx, (images, labels) in tqdm.tqdm(enumerate(test_loader),
                                                 total=len(test_loader),
                                                 desc="Attack Progress"):
        images = images.to(device)
        labels_np = labels.numpy()

        # Generate adversarial examples
        if targeted_class >= 0:
            target_labels = torch.full_like(labels, targeted_class).to(device)
            single_attack.set_mode_targeted_by_label(quiet=True)
            ensemble_ae = ensemble_attack(images, target_labels, targeted=True)
            single_ae = single_attack(images, target_labels)
        else:
            ensemble_ae = ensemble_attack(images, labels.to(device), targeted=False)
            single_ae = single_attack(images, labels.to(device))

        # Save some examples for visualization
        if batch_idx < 5:
            os.makedirs('adv_samples_cifar100', exist_ok=True)
            imshow(torchvision.utils.make_grid(ensemble_ae[:8]),
                   pth=f'adv_samples_cifar100/ensemble_{batch_idx}')

        # Evaluate attacks
        with torch.no_grad():
            # Ensemble attack evaluation
            ensemble_pred = final_model(ensemble_ae)
            ensemble_pred_np = ensemble_pred.argmax(dim=1).detach().cpu().numpy()
            ensemble_succ += (labels_np == ensemble_pred_np).sum()

            # Single attack evaluation
            single_pred = final_model(single_ae)
            single_pred_np = single_pred.argmax(dim=1).detach().cpu().numpy()
            single_succ += (labels_np == single_pred_np).sum()

        total += labels.shape[0]

    # Calculate and display results
    ensemble_asr = 100 * (1 - ensemble_succ / total)
    single_asr = 100 * (1 - single_succ / total)

    print("\n" + "="*60)
    print("FINAL RESULTS (CIFAR-100 with Soft Labels):")
    print("="*60)
    print(f"Ensemble Attack Success Rate (ASR): {ensemble_asr:.2f}%")
    print(f"Single Model Attack Success Rate (ASR): {single_asr:.2f}%")
    print(f"Improvement: {ensemble_asr - single_asr:.2f}%")
    print("="*60)

if __name__ == '__main__':
    main()