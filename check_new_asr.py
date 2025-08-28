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
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
import torchvision
import cv2
from PIL import Image
import torchattacks

import seaborn as sns
from sklearn.metrics import confusion_matrix

src_dic = "modsp"
device = "cuda:1"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def best_device():
    if torch.cuda.is_available():
        return 'cuda:0'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

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
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

def prepare_cifar10(num_clients=100, batch_size=64, device='cpu'):
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

def pgd(model, criterion, data, target, target_class=-1, epsilon=8/255., lr=2/255., num_iters=10):
    perturbations = (torch.rand_like(data, device=device) * 2 - 1) * epsilon
    target_adv = torch.tensor([target_class] * len(data), device=device).long()

    for it in range(num_iters):
        perturbations.requires_grad_()
        data_adv = torch.clamp(data + perturbations, 0, 1)
        output_adv = model(data_adv)
        # Ensure target is on the same device as output_adv
        target = target.to(output_adv.device)
        if target_class >= 0:  # Targeted attack
            loss = criterion(output_adv, target_adv)
        else:  # Untargeted attack
            loss = -criterion(output_adv, target)
        loss.backward()
        new_perturbations = perturbations - lr * perturbations.grad.sign()
        perturbations = torch.clamp(new_perturbations.detach(), -epsilon, epsilon)
    data_adv = torch.clamp(data + perturbations, 0, 1)
    return data_adv

def disjoint_test(m1, m2, test_loader):
    m1.to(device)
    m2.to(device)
    atk = torchattacks.PGD(m1, eps=8/255, alpha=2/255, steps=20)
    total_correct_clean = 0
    flipped = 0
    
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Get model2's predictions on clean images
        with torch.no_grad():
            clean_outputs = m2(images)
            clean_preds = clean_outputs.argmax(dim=1)
            correct_mask = (clean_preds == labels)
        
        # Generate adversarial images targeting model1
        adv_images = atk(images, labels)
        
        # Get model2's predictions on adversarial images
        with torch.no_grad():
            adv_outputs = m2(adv_images)
            adv_preds = adv_outputs.argmax(dim=1)
        
        # Update counts
        total_correct_clean += correct_mask.sum().item()
        flipped += (correct_mask & (adv_preds != labels)).sum().item()
    
    # Calculate ASR for model2
    asr2 = 100 * (flipped / total_correct_clean) if total_correct_clean > 0 else 0.0
    return f"{asr2:.2f}"

# ===============
# Main Evaluation
# ===============
def imshow(img, pth = "default"):
    # denormalize (approx)
    # img = img * 0.2 + 0.48

    npimg = img.cpu().detach().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(f'{pth}.png')
    plt.show()

def EPGDP(models, images, labels, targeted=False, epsilon=8/255., alpha=2/255., num_iters=20, random_start = True):
        images = images.clone().detach().to(device)
        labels = labels.clone().detach().to(device)

        loss = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()

        if random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(epsilon, epsilon)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(num_iters):
            adv_images.requires_grad = True

            ensemble_grad = torch.zeros_like(images).to(device)

            lom = models
            
            for model in lom:
                outputs = model(adv_images)
                # Calculate loss
                if targeted == True:
                    cost = -loss(outputs, labels)
                else:
                    cost = loss(outputs, labels)
                # Update adversarial images
                grad = torch.autograd.grad(
                    cost, adv_images, retain_graph=False, create_graph=False
                )[0]
                ensemble_grad += grad
            grad = ensemble_grad / len(models)

            adv_images = adv_images.detach() + alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-epsilon, max=epsilon)
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

class EnsembleMIFGSM:
    """
    Ensemble MI-FGSM using parameter perturbation for improved transferability.
    """

    def __init__(self, models, eps=8/255, alpha=2/255, steps=10, decay=1.0, device=device):
        self.models = models
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.decay = decay
        self.device = device

    def __call__(self, images, labels, targeted=False):
        """
        Perform ensemble MIFGSM attack.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        momentum = torch.zeros_like(images).detach().to(self.device)
        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True

            # Accumulate gradients from all models in ensemble
            ensemble_grad = torch.zeros_like(images).to(self.device)

            for model in self.models:
                # Get outputs from current model
                outputs = model(adv_images)

                # Calculate loss
                if targeted:
                    cost = -loss(outputs, labels)
                else:
                    cost = loss(outputs, labels)

                # Get gradient
                grad = torch.autograd.grad(
                    cost, adv_images, retain_graph=False, create_graph=False
                )[0]

                # Add to ensemble gradient
                ensemble_grad += grad

            # Average gradient across ensemble
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

def go_attack(ensemble_models, final_model, test_loader):

    targeted_class = -1

    final_model.to(device)

    noiters = 20
    epsilon = 8/255

    ensemble_attack = EnsembleMIFGSM(models=ensemble_models, eps=epsilon, alpha=2/255, steps=noiters, decay=1.0, device=device)
    total_correct_clean = 0
    flipped = 0
    
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Get model2's predictions on clean images
        with torch.no_grad():
            clean_outputs = final_model(images)
            clean_preds = clean_outputs.argmax(dim=1)
            correct_mask = (clean_preds == labels)
        
        if targeted_class >= 0:
            target_labels = torch.full_like(labels, targeted_class).to(device)
        else:
            target_labels = labels.to(device)
        
        # Generate adversarial images targeting model1
        adv_images = ensemble_attack(images, target_labels, targeted=(targeted_class >= 0))
        
        # Get model2's predictions on adversarial images
        with torch.no_grad():
            adv_outputs = final_model(adv_images)
            adv_preds = adv_outputs.argmax(dim=1)
        
        # Update counts
        total_correct_clean += correct_mask.sum().item()
        flipped += (correct_mask & (adv_preds != labels)).sum().item()
    
    # Calculate ASR for model2
    asr2 = 100 * (flipped / total_correct_clean) if total_correct_clean > 0 else 0.0
    return f"{asr2:.2f}"

def main():
    mix_method = 0 # 0 = left&right, 1 = overlay, 2 = ensemble

    targeted_class = -1  # Target class for the attack
    set_seed(2025)
    model, _, test_loader, classes = prepare_cifar10(batch_size=128, device=device)

    model_1 = torch.load(f'{src_dic}/100.pth')
    model_1.to(device)
    model_1.eval()

    model_2 = torch.load(f'{src_dic}/1000.pth')
    model_2.to(device)
    model_2.eval()

    model_3 = torch.load(f'exp_00_models_100/1.pth')
    model_3.to(device)
    model_3.eval()

    model_4 = torch.load(f'exp_00_models_100/2.pth')
    model_4.to(device)
    model_4.eval()

    print(f"ACC of original: {eval_model(model_1, test_loader)}")
    print(f"ACC of new new: {eval_model(model_3, test_loader)}")

    
    lom = [model_1, model_2, model_3, model_4]
    for i in range(0, len(lom)):
        loasr = []
        for j in range(0, len(lom)):
            loasr.append(disjoint_test(lom[i], lom[j], test_loader))
        print(loasr)
    
    ens = [model_3, model_4]
    print(go_attack(ens, model_2, test_loader))

if __name__ == '__main__':
    main()