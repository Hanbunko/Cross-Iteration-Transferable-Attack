import torch
import random
import numpy as np
import copy
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torchvision.models as models
from torch.functional import F
from torch.optim.lr_scheduler import ReduceLROnPlateau

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

dev = torch.device('cuda:0')

# CIFAR-10 dataset
tra_tran, tes_tran = [],[]
siz = 4
tra_tran += [transforms.RandomCrop(32, padding=siz), transforms.RandomHorizontalFlip()]
tra_tran += [transforms.ToTensor()]
tes_tran += [transforms.ToTensor()]
tra_set = datasets.CIFAR10('./data', train=True, download = True, transform = transforms.Compose(tra_tran))
tes_set = datasets.CIFAR10('./data', train=False, download = True, transform = transforms.Compose(tes_tran))

train_loader = DataLoader(tra_set, batch_size=64, shuffle=True)
test_loader = DataLoader(tes_set, batch_size=64, shuffle=False)
# federated setting
num_cli = 100
indices = list(range(len(tra_set)))
random.shuffle(indices)
loc_siz = len(indices) // num_cli
lotl = []
for idx in range(0, len(tra_set), loc_siz):
    lotl.append(DataLoader(Subset(tra_set, indices[idx:idx+loc_siz]),shuffle=True, batch_size=64))
ofs = 1000

model = ResNet18(num_classes=10).to(dev)
model = torch.load(f"./modsp/{ofs}.pth")
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= 200)
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
nor = 201
for r in range(1, nor):
    nop = int(num_cli * sam_rat)
    par = random.sample(range(num_cli), nop)
    accumulators = {key: torch.zeros_like(param, device=dev) for key, param in glomod.items()}
    for p in par:
        # load the global model
        model.load_state_dict(copy.deepcopy(glomod))
        # local train
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
    if r % 10 == 0:
        torch.save(model, f'./modsp/{r+ofs}.pth')
    for images, labels in test_loader:
        images, labels = images.to(dev), labels.to(dev)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct_clean += (predicted == labels).sum().item()
    print(f'FL Accuracy at round {r}: %.2f %%' % (100 * correct_clean / total))
