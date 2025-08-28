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

src_dic = "modsp"
device = "cuda:0"

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

def disjoint_test(m1, m2, test_loader):
    m1.to(device)
    m2.to(device)
    atk = torchattacks.PGD(m1, eps = 8/255, alpha = 2/255, steps = 20)
    succ1, succ2, total = 0,0,0
    for images, labels in test_loader:
        images = images.to(device)
        adv_images = atk(images, labels)
        pred1 = m1(adv_images)
        pred2 = m2(adv_images)
        succ1 += (labels.numpy() == pred1.argmax(dim=1).detach().cpu().numpy()).sum()
        succ2 += (labels.numpy() == pred2.argmax(dim=1).detach().cpu().numpy()).sum()
        total += labels.shape[0]
    asr1 = 100 * (1 - succ1 / total)
    asr2 = 100 * (1 - succ2 / total)
    # print(f"m1 ASR is {asr1}")
    # print(f"m2 ASR is {asr2}")
    return f"{asr2:.2f}"

leak_round = 100

noc = 1

model, trains, test_loader, classes = prepare_cifar10(num_clients = noc, batch_size=64, device=device)

model1 = torch.load(f'exp_01_models_2_100/1.pth')
model1.to(device)
model1.eval()

model2 = torch.load(f'exp_01_models_2_100/2.pth')
model2.to(device)
model2.eval()

model3 = torch.load(f'exp_01_models_2_100/3.pth')
model3.to(device)
model3.eval()

print(disjoint_test(model1, model2, test_loader))
print(disjoint_test(model1, model2, test_loader))

lom = [model1, model2]

targeted_class = -1
succ, total = 0,0
for batch_idx, [images, labels] in tqdm.tqdm(enumerate(test_loader)):
    images = images.to(device)

    new_labels = len(labels)* [targeted_class]  # Targeting class
    new_labels = torch.tensor(new_labels, device=device)
    ae = EPGDP(lom, images, labels, targeted = False, epsilon = 8/255, alpha = 2/255, num_iters = 20)

    pred = model1(ae)

    succ += (labels.numpy() == pred.argmax(dim=1).detach().cpu().numpy()).sum()
    # tar += (pred.argmax(dim=1).detach().cpu().numpy() == new_labels.cpu()).sum()
    total += labels.shape[0]
asr = 100 * (1 - succ / total)
print(asr)

succ, total = 0,0
for batch_idx, [images, labels] in tqdm.tqdm(enumerate(test_loader)):
    images = images.to(device)

    new_labels = len(labels)* [targeted_class]  # Targeting class
    new_labels = torch.tensor(new_labels, device=device)
    ae = EPGDP(lom, images, labels, targeted = False, epsilon = 8/255, alpha = 2/255, num_iters = 20)

    pred = model2(ae)

    succ += (labels.numpy() == pred.argmax(dim=1).detach().cpu().numpy()).sum()
    # tar += (pred.argmax(dim=1).detach().cpu().numpy() == new_labels.cpu()).sum()
    total += labels.shape[0]
asr = 100 * (1 - succ / total)
print(asr)

succ, total = 0,0
for batch_idx, [images, labels] in tqdm.tqdm(enumerate(test_loader)):
    images = images.to(device)

    new_labels = len(labels)* [targeted_class]  # Targeting class
    new_labels = torch.tensor(new_labels, device=device)
    ae = EPGDP(lom, images, labels, targeted = False, epsilon = 8/255, alpha = 2/255, num_iters = 20)

    pred = model3(ae)

    succ += (labels.numpy() == pred.argmax(dim=1).detach().cpu().numpy()).sum()
    # tar += (pred.argmax(dim=1).detach().cpu().numpy() == new_labels.cpu()).sum()
    total += labels.shape[0]
asr = 100 * (1 - succ / total)
print(asr)