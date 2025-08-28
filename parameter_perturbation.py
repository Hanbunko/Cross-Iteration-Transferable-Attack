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

# ===============================
# Parameter Perturbation Functions
# ===============================
def create_diversified_ensemble(early_model, train_set, num_models=10, noise_configs=None):
    """
    Create an ensemble of diversified models using layer-specific noise scales.
    """
    if noise_configs is None:
        # Default noise configurations based on research
        noise_configs = {
            'shortcut': 0.07,  # Highest for skip connections
            'bn': 0.06,        # High for batch normalization
            'conv': 0.05,      # Medium for convolution
            'linear': 0.02     # Lowest for final linear layer
        }

    ensemble_models = [early_model]  # Include original model

    for i in range(num_models - 1):
        perturbed_model = copy.deepcopy(early_model)

        with torch.no_grad():
            for name, param in perturbed_model.named_parameters():
                if not param.requires_grad:
                    continue

                # Determine noise scale based on layer type
                if 'shortcut' in name:
                    noise_scale = noise_configs.get('shortcut', 0.04)
                elif 'bn' in name:
                    noise_scale = noise_configs.get('bn', 0.03)
                elif 'conv' in name:
                    noise_scale = noise_configs.get('conv', 0.02)
                elif 'linear' in name:
                    noise_scale = noise_configs.get('linear', 0.01)
                else:
                    noise_scale = 0.01

                # Add normalized noise
                if torch.std(param) > 0:
                    noise = torch.randn_like(param) * noise_scale * torch.std(param)
                    param.add_(noise)

        ensemble_models.append(perturbed_model)

    return ensemble_models

# ===============================
# Ensemble MIFGSM Attack
# ===============================
class EnsembleMIFGSM:
    """
    Ensemble MI-FGSM using parameter perturbation for improved transferability.
    """

    def __init__(self, models, eps=8/255, alpha=2/255, steps=10, decay=1.0, device='cuda:0'):
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
                    cost, adv_images, retain_graph=True, create_graph=False
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

def eval_model(m, test_loader):
    succ, total = 0,0
    for images, labels in test_loader:
        images = images.to(device)
        pred = m(images)
        succ += (labels.numpy() == pred.argmax(dim=1).detach().cpu().numpy()).sum()
        total += labels.shape[0]
    acc = 100 * (succ / total)
    return acc

def disjoint_test(m1, m2, test_loader):
    print("AE is generated by m1")
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
    print(f"m1 ASR is {asr1}")
    print(f"m2 ASR is {asr2}")

# ===============
# Main Evaluation
# ===============
def main():
    targeted_class = -1  # Target class for the attack (-1 for untargeted)
    num_ensemble_models = 10  # Number of models in ensemble

    set_seed(2025)
    model, trains, test_loader, classes = prepare_cifar10(batch_size=64, device=device)

    lol = [0]*10
    for images, labels in trains[0]:
        for l in labels:
            lol[l] += 1
    print(lol)

    # Load final model for evaluation
    final_model = torch.load(f'{src_dic}/1000.pth')
    final_model.to(device)
    final_model.eval()

    print("="*60)
    tasrs = []
    asrs = []

    # Results storage
    ensemble_results = []
    single_results = []

    rounds = list(range(100, 200, 100))

    for r in rounds:
        print(f"Doing Round {r}...")
        early_model = torch.load(f'{src_dic}/{r}.pth')
        early_model.to(device)
        early_model.eval()

        # Create diversified ensemble
        print(f"Creating ensemble of {num_ensemble_models} models...")

        level = 0
        if level == 0:
            nc = { 'shortcut': 0.07, 'bn': 0.06, 'conv': 0.05, 'linear': 0.02}
        if level == 1:
            nc = { 'shortcut': 0.08, 'bn': 0.08, 'conv': 0.08, 'linear': 0.03}
        ensemble_models = create_diversified_ensemble(early_model=early_model, train_set=trains[0], num_models=num_ensemble_models, noise_configs=nc)

        disjoint_test(ensemble_models[1], ensemble_models[2], test_loader)

        # accuracy things
        loacc = []
        oriacc = eval_model(early_model, test_loader)
        for mod in ensemble_models:
            acccc = eval_model(mod, test_loader)
            loacc.append(acccc)
        print(f"Original ACC: {oriacc}; Perturbed avg ACC: {sum(loacc)/len(loacc)}")

        ensemble_attack = EnsembleMIFGSM(
            models=ensemble_models,
            eps=8/255,
            alpha=2/255,
            steps=20,
            decay=1.0,
            device=device
        )
        single_attack = torchattacks.MIFGSM(
            early_model,
            eps=8/255,
            alpha=2/255,
            steps=20
        )

        total = 0
        ensemble_succ, ensemble_tar = 0, 0
        single_succ, single_tar = 0, 0

        for batch_idx, [images, labels] in tqdm.tqdm(enumerate(test_loader)):
            images = images.to(device)
            labels_np = labels.numpy()

            # Create target labels
            if targeted_class >= 0:
                target_labels = torch.full_like(labels, targeted_class).to(device)
                single_attack.set_mode_targeted_by_label(quiet = True)
            else:
                target_labels = labels.to(device)

            ensemble_ae = ensemble_attack(
                images,
                target_labels,
                targeted=(targeted_class >= 0)
            )
            if targeted_class >= 0:
                single_ae = single_attack(images, target_labels)
            else:
                single_ae = single_attack(images, labels.to(device))

            # Evaluate ensemble attack
            with torch.no_grad():
                ensemble_pred = final_model(ensemble_ae)
                ensemble_pred_np = ensemble_pred.argmax(dim=1).detach().cpu().numpy()

                ensemble_succ += (labels_np == ensemble_pred_np).sum()
                if targeted_class >= 0:
                    ensemble_tar += (ensemble_pred_np == targeted_class).sum()

                # Evaluate single attack
                single_pred = final_model(single_ae)
                single_pred_np = single_pred.argmax(dim=1).detach().cpu().numpy()

                single_succ += (labels_np == single_pred_np).sum()
                if targeted_class >= 0:
                    single_tar += (single_pred_np == targeted_class).sum()

            total += labels.shape[0]

        # Calculate metrics
        if targeted_class >= 0:
            ensemble_tasr = 100 * (ensemble_tar / total)
            single_tasr = 100 * (single_tar / total)
            tasrs.append((ensemble_tasr, single_tasr))
            print(f"Round {r} - Ensemble TASR: {ensemble_tasr:.2f}%, Single TASR: {single_tasr:.2f}%")

        ensemble_asr = 100 * (1 - ensemble_succ / total)
        single_asr = 100 * (1 - single_succ / total)
        asrs.append((ensemble_asr, single_asr))
        print(f"Round {r} - Ensemble ASR: {ensemble_asr:.2f}%, Single ASR: {single_asr:.2f}%")

        ensemble_results.append(ensemble_asr)
        single_results.append(single_asr)

    print("\n" + "="*60)
    print("Summary Results:")
    print("="*60)

    if targeted_class >= 0:
        print("\nTargeted Attack Success Rates (TASR):")
        for i, r in enumerate(rounds):
            print(f"Round {r}: Ensemble={tasrs[i][0]:.2f}%, Single={tasrs[i][1]:.2f}%")

    print("\nAttack Success Rates (ASR):")
    for i, r in enumerate(rounds):
        print(f"Round {r}: Ensemble={asrs[i][0]:.2f}%, Single={asrs[i][1]:.2f}%")

    # Calculate improvement
    avg_ensemble = np.mean(ensemble_results)
    avg_single = np.mean(single_results)

    print(f"\nAverage Ensemble ASR: {avg_ensemble:.2f}%")
    print(f"Average Single ASR: {avg_single:.2f}%")
    print("="*60)

if __name__ == '__main__':
    main()