import torch
import random
import numpy as np
import copy
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.utils
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torchvision.models as models
from torch.functional import F
import os

import seaborn as sns
from sklearn.metrics import confusion_matrix

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

see = 314159265
random.seed(see)
np.random.seed(see)
torch.manual_seed(see)

loc = ['0-airplane', '1-automobile', '2-bird', '3-cat','4-deer', '5-dog', '6-frog', '7-horse', '8-ship', '9-truck']

dev = torch.device('cuda')

tra_tran, tes_tran = [], []
mea = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)
siz = 4
tra_tran += [transforms.RandomCrop(32,  padding=siz), transforms.RandomHorizontalFlip()]
tra_tran += [transforms.ToTensor()]
tes_tran += [transforms.ToTensor()]
norm = transforms.Normalize(mea, std)
# tra_tran += [norm]
# tes_tran += [norm]
tra_set = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.Compose(tra_tran))
tes_set = datasets.CIFAR10('./data', train=False, download=True, transform=transforms.Compose(tes_tran))

train_loader = DataLoader(tra_set, batch_size=64, shuffle=True)
test_loader = DataLoader(tes_set, batch_size=64, shuffle=False)

model = ResNet18(num_classes=10).to(dev)
model.eval()

fin = copy.deepcopy(model)
fin = torch.load('modsp/1000.pth')
fin = fin.to(dev)
fin.eval()

classes = tra_set.classes

def pgd_attack(model, images, labels, eps=0.031, alpha=0.003, iters=10):
    images = images.clone().detach().to(dev)
    labels = labels.to(dev)
    ori_images = images.clone().detach()

    for i in range(iters):
        images.requires_grad = True
        outputs = model(images)
        model.zero_grad()
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()

        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach()
    return images, eta

def targeted_pgd_attack(model, images, target_labels, eps=0.031, alpha=0.003, iters=10):
    images = images.clone().detach().to(dev)
    target_labels = target_labels.to(dev)
    ori_images = images.clone().detach()

    for i in range(iters):
        images.requires_grad = True
        outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, target_labels)
        model.zero_grad()
        loss.backward()
        adv_images = images - alpha * images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach()
    return images, eta


def plot_confusion_matrix(predictions, labels, rou, tpy, class_names=None):
    # Create the cmp directory if it doesn't exist
    os.makedirs('./cmp', exist_ok=True)
    
    # Convert tensors to numpy arrays
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()

    cm = confusion_matrix(labels, predictions)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')

    # Add labels
    if class_names is None:
        class_names = [str(i) for i in range(len(cm))]

    plt.xlabel('Predicted')
    plt.ylabel('Clean')
    if tpy == -2:
        plt.title(f'(clean vs. predicted) Confusion Matrix (Clean) at round {rou}')
    elif tpy == -1:
        plt.title(f'(clean vs. predicted) Confusion Matrix (Untargeted PGD) at round {rou}')
    else:
        plt.title(f'(clean vs. predicted) Confusion Matrix (Targeted PGD toward {tpy}) at round {rou}')

    # Add class names
    plt.xticks(np.arange(len(class_names)) + 0.5, class_names, rotation=90)
    plt.yticks(np.arange(len(class_names)) + 0.5, class_names, rotation=0)

    plt.tight_layout()
    print(f'Saving confusion matrix to ./cmp/cvp_confusion_matrix_{rou}_{tpy}.png')
    plt.savefig(f'./cmp/cvp_confusion_matrix_{rou}_{tpy}.png')
    plt.close()

x = range(200,1001,400)
for i in range(len(x)):
    r = x[i]
    model = torch.load(f'modsp/{r}.pth')
    model.to(dev)
    model.eval()

    # clean case
    all_predictions = []
    all_labels = []
    for images, labels in test_loader:
        images, labels = images.to(dev), labels.to(dev)
        ori = model(images)
        _, predicted = torch.max(ori.data, 1)
        # Append batch predictions and labels to our lists
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    plot_confusion_matrix(np.array(all_predictions), np.array(all_labels), rou = r, tpy = -2, class_names=loc)

    # untargeted pgd
    all_predictions = []
    all_labels = []
    all_clean = []
    
    for images, labels in test_loader:
        images, labels = images.to(dev), labels.to(dev)
        adv_images,ptb = pgd_attack(model, images, labels)
        ori = fin(images)
        outputs = fin(adv_images)
        _, oripre = torch.max(ori.data, 1)
        _, predicted = torch.max(outputs.data, 1)
        # Append batch predictions and labels to our lists
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_clean.extend(oripre.cpu().numpy())
    
    plot_confusion_matrix(np.array(all_predictions), np.array(all_clean), rou = r, tpy = -1, class_names=loc)

    # targeted pgd
    for k in range(10):
        all_predictions = []
        all_labels = []
        all_clean = []
        for images, labels in test_loader:
            images, labels = images.to(dev), labels.to(dev)
            target_labels = torch.full_like(labels, k).to(dev)
            adv_images,ptb = targeted_pgd_attack(model, images, target_labels)
            ori = fin(images)
            outputs = fin(adv_images)
            _, oripre = torch.max(ori.data, 1)
            _, predicted = torch.max(outputs.data, 1)
            # Append batch predictions and labels to our lists
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_clean.extend(oripre.cpu().numpy())
        plot_confusion_matrix(np.array(all_predictions), np.array(all_clean), rou = r, tpy = k, class_names=loc)

"""
x = range(0,1100,100)
for k in range(0,10):
    loars = []
    lll = []
    for i in range(len(x)):
        r = x[i]
        model = torch.load(f'modsp/{r}.pth')
        model.to(dev)
        model.eval()

        suc_tpgd = 0
        conp = 0
        correct = 0
        ttt = 0
        for images, labels in test_loader:
            images, labels = images.to(dev), labels.to(dev)
            tc = k
            target_labels = torch.full_like(labels, tc).to(dev)
            adv_images,ptb = targeted_pgd_attack(model, images, target_labels)
            ori = fin(images)
            outputs = fin(adv_images)
            _, oripre = torch.max(ori.data, 1)
            _, predicted = torch.max(outputs.data, 1)
            correct += (oripre == labels).sum().item()
            con = (predicted == target_labels) & (oripre == labels)
            conp += (predicted == target_labels).sum().item()
            ttt += (labels == labels).sum().item()
            suc_tpgd += con.sum().item()
        loars.append(100 * suc_tpgd / correct)
        lll.append(100 * conp/ttt)
        print(f'targeted PGD toward {tc}: %.2f %%' % (100 * conp/ttt))
        print(f'targeted PGD toward {tc}: %.2f %%' % (100 * suc_tpgd / correct))
    print(loars)
    print(ttt)
"""

