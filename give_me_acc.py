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

def prepare_cifar10(num_clients = 100, batch_size=64, device='cpu'):
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
def check_pgd(mod, loa, tc):
    """
    Fixed function to check PGD attack effectiveness
    """
    mod.eval()
    correct_original = 0
    correct_target = 0
    total = 0
    lop = []

    with torch.no_grad():
        for images, labels in loa:
            images, labels = images.to(device), labels.to(device)
            outputs = mod(images)
            _, predicted = outputs.max(1)

            # Count how many are correctly classified (should be low for good adversarial examples)
            correct_original += predicted.eq(labels).sum().item()

            # Count how many are classified as the target class
            target_labels = torch.where(labels == tc,
                                       torch.ones_like(labels) * ((tc - 1) % 10),
                                       torch.ones_like(labels) * tc)
            correct_target += predicted.eq(target_labels).sum().item()

            total += labels.size(0)
            for idx in range(len(predicted)):
                lop.append((labels[idx].item(), predicted[idx].item()))

    orig_acc = 100. * correct_original / total
    target_acc = 100. * correct_target / total
    print(f"  PGD Check - Original Acc: {orig_acc:.2f}% | Target Class is {tc} Acc: {target_acc:.2f}%")
    print(lop)
    return orig_acc, target_acc
# ===============================
# Enhanced Ensemble MIFGSM Attack
# ===============================
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
    
class EnsembleMIFGSMWeight:
    """
    Ensemble MI-FGSM using parameter perturbation for improved transferability.
    """

    def __init__(self, models, eps=8/255, alpha=2/255, steps=10, decay=1.0, device=device, weights = None):
        self.models = models
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.decay = decay
        self.device = device
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            assert len(weights) == len(models)
            total_weight = sum(weights)
            self.weights = [w / total_weight for w in weights]

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

            for i, model in enumerate(self.models):
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
                ensemble_grad += self.weights[i] * grad.detach()

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
        m.to(device)
        pred = m(images)
        succ += (labels.numpy() == pred.argmax(dim=1).detach().cpu().numpy()).sum()
        total += labels.shape[0]
    acc = 100 * (succ / total)
    return acc

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

def load_ensembles(early_model, num_models, path, num_cli):
    ensemble_models = [early_model]
    for idx in range(num_models):
        midx = idx+1
        model = torch.load(f"./{path}/{midx}.pth")
        ensemble_models.append(model)
    print(f"{len(ensemble_models)} HEROS ENSEMBLED!")
    return ensemble_models

def imshow(img, pth = "default"):
    npimg = img.cpu().detach().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(f'{pth}.png')
    plt.show()

def check_epsilon(img1, img2):
    for i in range(0,3,1):
        for j in range (0,32,1):
            for k in range(0,32,1):
                d = img1[i][j][k] - img2[i][j][k]
                if torch.abs(d) > 0.0313726:
                    torch.set_printoptions(precision = 8)
                    print(img1[i][j][k])
                    print(img2[i][j][k])
                    print(d)
                    return False
    return True

# ===============
# Main Evaluation
# ===============
def main():

    targeted_class = -1
    num_ensemble_models = 200
    method = 4
    noc = 1

    mod_dir = "exp_01_models_3_100"
    show_matrix = False
    att_all = True

    print(f"Number of Client is {noc}")
    print(f"Number of New Models is {num_ensemble_models}")
    if method == 0:
        print("Using the very first method")
    elif method == 1:
        print("Using all-previous-ensemble")
    elif method == 2:
        print("Using SUB-previous-ensemble")
    elif method == 3:
        print("Using SUB-2-previous-ensemble")
    else:
        print("Using Distillation")


    set_seed(2025)
    model, trains, test_loader, classes = prepare_cifar10(num_clients = noc, batch_size=64, device=device)

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

    rounds = list(range(900, 1000, 10))

    for r in rounds:
        print(f"\nEvaluating Round {r}...")
        early_model = torch.load(f'{src_dic}/{r}.pth')
        early_model.to(device)
        early_model.eval()

        # Evaluate ensemble quality
        loacc = []
        oriacc = eval_model(early_model, test_loader)
        print(f"Model at Round {r} ACC: {oriacc:.2f}%")

        continue

        for i, mod in enumerate(ensemble_models):
            acccc = eval_model(mod, test_loader)
            loacc.append(acccc)


        print(f"Ensemble Average ACC: {sum(loacc)/len(loacc):.2f}%")
        print(f"Original Model ACC: {oriacc:.2f}%")

        noiters = 20
        epsilon = 8/255

        # Initialize enhanced ensemble attack
        ensemble_attack = EnsembleMIFGSM(models=ensemble_models, eps=epsilon, alpha=2/255, steps=noiters, decay=1.0, device=device)
        #www = [1.15,1,1.2,1.15,1,1.2,1.15,1,1.2,1.15,1]
        #ensemble_attack = EnsembleMIFGSMWeight(models=ensemble_models, eps=epsilon, alpha=2/255, steps=noiters, decay=1.0, device=device,weights = www)

        # Standard single model attack for comparison
        single_attack = torchattacks.MIFGSM(early_model, eps=epsilon, alpha=2/255, steps=noiters)

        total = 0
        ensemble_succ, ensemble_tar = 0, 0
        single_succ, single_tar = 0, 0

        vvv = [0]*(len(ensemble_models))
        aaa = [0]*(len(ensemble_models))

        print("\nGenerating adversarial examples...")
        for batch_idx, [images, labels] in tqdm.tqdm(enumerate(test_loader),
                                                    total=len(test_loader),
                                                    desc="Attack Progress"):
            images = images.to(device)
            labels_np = labels.numpy()

            # Create target labels
            if targeted_class >= 0:
                target_labels = torch.full_like(labels, targeted_class).to(device)
                single_attack.set_mode_targeted_by_label(quiet = True)
            else:
                target_labels = labels.to(device)

            # Generate adversarial examples
            ensemble_ae = ensemble_attack(
                images,
                target_labels,
                targeted=(targeted_class >= 0)
            )

            
            i = 0
            if att_all == True:
                for mmm in ensemble_models:
                    with torch.no_grad():
                        ppp = mmm(ensemble_ae)
                        pppc = ppp.argmax(dim=1).detach().cpu().numpy()
                        vvv[i] += (labels_np == pppc).sum()
                    aaa[i] += labels.shape[0]
                    i = i+1

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
        print(f"Improvement: {ensemble_asr - single_asr:.2f}%")

        ensemble_results.append(ensemble_asr)
        single_results.append(single_asr)
    
    return 0

    print("\n" + "="*60)
    print("Summary Results:")
    print("="*60)
    if att_all == True:
        print(vvv)
        print(f"Mean: {100 - np.mean(vvv)/100: .2f}%")

    if targeted_class >= 0:
        print("\nTargeted Attack Success Rates (TASR):")
        for i, r in enumerate(rounds):
            improvement = tasrs[i][0] - tasrs[i][1]
            print(f"Round {r}: Ensemble={tasrs[i][0]:.2f}%, Single={tasrs[i][1]:.2f}%, Improvement={improvement:.2f}%")

    print("\nAttack Success Rates (ASR):")
    for i, r in enumerate(rounds):
        improvement = asrs[i][0] - asrs[i][1]
        print(f"Round {r}: Ensemble={asrs[i][0]:.2f}%, Single={asrs[i][1]:.2f}%, Improvement={improvement:.2f}%")

if __name__ == '__main__':
    main()