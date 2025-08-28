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
        return 'cuda:1'
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

# ===============
# Main Evaluation
# ===============
def plot_confusion_matrix(predictions, labels, rou, tpy, class_names=None):
    # Create the cmp directory if it doesn't exist
    os.makedirs('./vnites', exist_ok=True)
    
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
    plt.savefig(f'./vnites/cvp_confusion_matrix_{rou}_{tpy}.png')
    plt.close()

def imshow(img, pth = "default"):
    # denormalize (approx)
    # img = img * 0.2 + 0.48

    npimg = img.cpu().detach().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(f'{pth}.png')
    plt.show()

def main():
    targeted_class = -1  # Target class for the attack
    set_seed(2025)
    device = best_device()
    model, _, test_loader, classes = prepare_cifar10(batch_size=128, device=device)

    final_model = torch.load(f'{src_dic}/1000.pth')
    final_model.to(device)
    final_model.eval()
    criterion = nn.CrossEntropyLoss()

    print("="*60)
    tasrs = []
    asrs = []
    succ, tar, total = 0,0,0
    rounds = list(range(100, 1100, 100))
    for r in rounds:
        early_model = torch.load(f'{src_dic}/{r}.pth')
        early_model.to(device)
        early_model.eval()
        for batch_idx, [images, labels] in tqdm.tqdm(enumerate(test_loader)):
            images = images.to(device)

            atk = torchattacks.VNIFGSM(early_model, eps = 8/255, alpha = 2/255, steps = 10, N=20)
            new_labels = len(labels)* [targeted_class]  # Targeting class
            new_labels = torch.tensor(new_labels, device=device)
            if targeted_class >= 0:
                atk.set_mode_targeted_by_label()
                ae = atk(images, new_labels)
            else:
                ae = atk(images, labels)
            pred = final_model(ae)
            succ += (labels.numpy() == pred.argmax(dim=1).detach().cpu().numpy()).sum()
            tar += (pred.argmax(dim=1).detach().cpu().numpy() == new_labels.cpu()).sum()
            total += labels.shape[0]

            torch.save(ae[1], f'sam_mim/{batch_idx}.pt')
            imshow(torchvision.utils.make_grid(ae[1]), pth = f'sam_mim/{batch_idx}')
        if targeted_class >= 0:
            tasr = 100 * (tar / total)
            tasrs.append(tasr)
        asr = 100 * (1 - succ / total)
        asrs.append(asr)
    print(tasrs)
    print(asrs)
    print("="*60)

if __name__ == '__main__':
    main()