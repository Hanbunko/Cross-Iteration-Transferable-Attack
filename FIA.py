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
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from collections import Counter
from PIL import Image
import torchattacks

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
        self.proj2d = nn.Linear(512*block.expansion, 2)  # 2D projection layer
        self.linear = nn.Linear(2, num_classes)          # Final classifier now takes 2D input

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_features=False, return_2d=False):
        out1 = F.relu(self.bn1(self.conv1(x)))
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        out6 = F.avg_pool2d(out5, 4)
        out7 = out6.view(out6.size(0), -1)
        out2d = self.proj2d(out7)  # 2D features
        out8 = self.linear(out2d)  # Class logits

        if return_2d:
            return out2d, out8
        if return_features:
            return out1, out2, out3, out4, out5, out6, out7, out8
        return out8


def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


dev = torch.device('cuda')

# CIFAR-10 dataset
tra_tran, tes_tran = [],[]
mea = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)
siz = 4
tra_tran += [transforms.RandomCrop(32, padding=siz), transforms.RandomHorizontalFlip()]
tra_tran += [transforms.ToTensor()]
tes_tran += [transforms.ToTensor()]
norm = transforms.Normalize(mea, std)
# tra_tran += [norm]
# tes_tran += [norm]

# Filter CIFAR-10 to only include classes 1, 5, and 6, and relabel them to 0, 1, 2
class_map = {0: 0, 1: 1, 2: 2}

class FilteredCIFAR10(torch.utils.data.Dataset):
    def __init__(self, cifar_dataset, class_map):
        self.data = []
        self.targets = []
        for img, label in zip(cifar_dataset.data, cifar_dataset.targets):
            if label in class_map:
                self.data.append(img)
                self.targets.append(class_map[label])
        self.data = np.array(self.data)
        self.targets = np.array(self.targets)
        self.transform = cifar_dataset.transform
    def __len__(self):
        return len(self.targets)
    def __getitem__(self, idx):
        img, label = self.data[idx], self.targets[idx]
        # Convert numpy image to PIL Image for transform
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, label

# Replace tra_set and tes_set with filtered datasets
tra_set = FilteredCIFAR10(datasets.CIFAR10('./data', train=True, download=True, transform=transforms.Compose(tra_tran)), class_map)
tes_set = FilteredCIFAR10(datasets.CIFAR10('./data', train=False, download=True, transform=transforms.Compose(tes_tran)), class_map)

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

ofs = 0

model = ResNet18(num_classes=3).to(dev)
# model = torch.load(f"./flm/{ofs}.pth")
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= 200)
criterion = nn.CrossEntropyLoss()

# really init model
torch.save(model, f'./mod012/init.pth')

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
nor = 0
for r in range(nor):
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
        torch.save(model, f'./mod012/{r+ofs}.pth')
    for images, labels in test_loader:
        images, labels = images.to(dev), labels.to(dev)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct_clean += (predicted == labels).sum().item()
    print(f'FL Accuracy at round {r}: %.2f %%' % (100 * correct_clean / total))

def plot_decision_boundary(model, loader, device, round_num, save_path=None):
    model.eval()
    features_2d = []
    labels_list = []
    preds_list = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            out2d, logits = model(images, return_2d=True)
            features_2d.append(out2d.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
            preds_list.append(torch.argmax(logits, dim=1).cpu().numpy())
    features_2d = np.concatenate(features_2d, axis=0)
    labels_list = np.concatenate(labels_list, axis=0)
    preds_list = np.concatenate(preds_list, axis=0)

    # Dynamically determine present classes
    present_classes = np.unique(labels_list)
    n_classes = len(present_classes)
    # Use a visually distinct colormap for up to 3 classes
    custom_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    cmap = ListedColormap(custom_colors[:n_classes])

    # Map relabeled indices back to original labels for legend
    relabel_to_original = {0: 0, 1: 1, 2: 2}

    plt.figure(figsize=(8, 8))
    # Plot decision boundary
    x_min, x_max = features_2d[:, 0].min() - 1, features_2d[:, 0].max() + 1
    y_min, y_max = features_2d[:, 1].min() - 1, features_2d[:, 1].max() + 1
    # xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    xx, yy = np.meshgrid(np.linspace(-40, 40, 300), np.linspace(-40, 40, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.tensor(grid, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model.linear(grid_tensor)
        Z = torch.argmax(logits, dim=1).cpu().numpy()
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, levels=np.arange(n_classes+1)-0.5, alpha=0.2, cmap=cmap)

    # Plot the 2D features colored by class, using the same cmap
    for i, cls in enumerate(present_classes):
        idx = labels_list == cls
        orig_label = relabel_to_original.get(cls, cls)
        plt.scatter(features_2d[idx, 0], features_2d[idx, 1], label=f'Class {orig_label}', alpha=0.5, s=10, color=cmap(i))
    plt.legend()
    plt.title(f'Decision Boundary at Round {round_num}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    if save_path:
        plt.savefig(save_path)
    plt.show()

def fia(model, criterion, data, target, target_class=-1, epsilon=8/255., lr=0.8/255., num_iters=30, feature_layer=None):
    """
    Feature Importance-aware Attack (FIA)
    """
    device = data.device
    if feature_layer is None:
        # Default to layer4, which is a good choice for ResNet feature extraction
        feature_layer = model.layer4

    # --- 1. Get feature importance ---
    feature_gradients = []
    def hook(module, grad_in, grad_out):
        feature_gradients.append(grad_out[0])
    
    handle = feature_layer.register_full_backward_hook(hook)

    model.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    
    handle.remove()
    
    if not feature_gradients:
        raise ValueError("Could not get feature gradients. Check feature_layer.")

    grad = feature_gradients[0]
    # Importance is the mean absolute gradient over batch, height, and width for each channel
    importance = torch.mean(torch.abs(grad), dim=[0, 2, 3])
    
    # --- 2. Generate adversarial example ---
    perturbations = (torch.rand_like(data, device=device) * 2 - 1) * epsilon
    target_adv = torch.tensor([target_class] * len(data), device=device).long()
    
    # Get original features to use in the FIA loss
    with torch.no_grad():
        original_features = []
        def forward_hook(module, input, output):
            original_features.append(output)
        
        hook_handle = feature_layer.register_forward_hook(forward_hook)
        model(data)
        hook_handle.remove()
        original_feature_map = original_features[0].detach()

    for it in range(num_iters):
        perturbations.requires_grad_()
        data_adv = torch.clamp(data + perturbations, 0, 1)

        # We need to get the feature map of the adversarial example
        adv_features = []
        def adv_forward_hook(module, input, output):
            adv_features.append(output)
        
        hook_handle = feature_layer.register_forward_hook(adv_forward_hook)
        output_adv = model(data_adv)
        hook_handle.remove()
        adv_feature_map = adv_features[0]

        # FIA loss
        # classification loss
        if target_class >= 0:
            loss_cls = criterion(output_adv, target_adv)
        else: # untargeted
            loss_cls = -criterion(output_adv, target) # maximize loss for original class

        # feature loss (weighted L2)
        feature_diff = adv_feature_map - original_feature_map
        loss_feat = torch.sum(importance.view(1, -1, 1, 1) * (feature_diff ** 2))
        
        # A-FIA uses a hyperparameter to balance. We'll use a fixed one.
        loss_fia = loss_cls + 1e-5 * loss_feat
        
        loss_fia.backward()
        
        # PGD step
        with torch.no_grad():
            if target_class >= 0: # targeted: gradient descent to minimize loss
                new_perturbations = perturbations - lr * perturbations.grad.sign()
            else: # untargeted: gradient ascent to maximize loss
                new_perturbations = perturbations + lr * perturbations.grad.sign()
            
            perturbations = torch.clamp(new_perturbations, -epsilon, epsilon)
    
    data_adv = torch.clamp(data + perturbations, 0, 1)
    return data_adv

def pgd(model, criterion, data, target, target_class=-1, epsilon=8/255., lr=0.8/255., num_iters=30):
    device = data.device

    perturbations = (torch.rand_like(data, device=device) * 2 - 1) * epsilon
    target_adv = torch.tensor([target_class] * len(data), device=device).long()

    for it in range(num_iters):
        perturbations.requires_grad_()
        data_adv = torch.clamp(data + perturbations, 0, 1)
        output_adv = model(data_adv)
        if target_class >= 0:  # Targeted attack
            loss = criterion(output_adv, target_adv)
        else:  # Untargeted attack
            loss = -criterion(output_adv, target)
        loss.backward()
        new_perturbations = perturbations - lr * perturbations.grad.sign()
        perturbations = torch.clamp(new_perturbations.detach(), -epsilon, epsilon)
    data_adv = torch.clamp(data + perturbations, 0, 1)
    return data_adv

def rescale(img):
    cp = img.clone()
    return (cp - cp.min()) / (cp.max() - cp.min() + 1e-8)

if __name__ == "__main__":
    attack_method = 'fia'  # Options: 'pgd', 'fia', 'cw', 'apgd'

    # Set seed for reproducible selection of the original image
    seed = 2025
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    ori_cla = 2
    label0_indices = [i for i, t in enumerate(tes_set.targets) if t == ori_cla]
    idx = random.choice(label0_indices)
    orig_img, orig_label = tes_set[idx]
    orig_img = orig_img.unsqueeze(0).to(dev)
    orig_label = torch.tensor([orig_label], device=dev)

    # Remove/reset seed so PGD attack is not deterministic
    random.seed()
    np.random.seed()
    torch.manual_seed(torch.initial_seed())

    # 2. Load model at round 100
    model_100 = torch.load('./mod012/100.pth', map_location=dev)
    model_100.eval()

    # 3. Generate 1000 different adversarial examples
    target_class = 0
    noae = 1000
    adv_imgs = []
    for i in range(noae):
        # Each time, the attack will be different due to random start & seed reset
        if i % 100 == 0:
            print(f'{i} out of {noae} adversarial examples generated.')
        if attack_method == 'pgd':
            adv_img = pgd(model_100, criterion, orig_img, orig_label, target_class=target_class)
        elif attack_method == 'fia':
            adv_img = fia(model_100, criterion, orig_img, orig_label, target_class=target_class, feature_layer=model_100.layer4)
        elif attack_method == 'cw':
            atk = torchattacks.CW(model_100, c=0.1, kappa=0, steps=100, lr=0.01)
            if target_class < 0:
                raise ValueError("C&W attack in this script is configured for targeted attacks. Please set a target_class >= 0.")
            atk.set_mode_targeted_by_label(quiet=True)
            target_labels = torch.tensor([target_class] * len(orig_img), device=dev)
            adv_img = atk(orig_img, orig_label)
        elif attack_method == 'apgd':
            atk = torchattacks.APGD(model_100, eps = 8/255, seed = i+1000)
            adv_img = atk(orig_img, orig_label)
        else:
            raise ValueError(f"Unknown attack method: {attack_method}")
        adv_imgs.append(adv_img)

    # 4. Get 2D feature for the original point
    with torch.no_grad():
        orig_feat2d, _ = model_100(orig_img, return_2d=True)
    orig_feat2d = orig_feat2d.cpu().numpy()[0]

    # 5. Visualize decision boundary and overlay the original and 100 perturbed points and lines at multiple rounds
    rounds = [100, 200, 500, 1000, 2000]
    for r in rounds:
        model_r = torch.load(f'./mod012/{r}.pth', map_location=dev)
        model_r.eval()
        features_2d = []
        labels_list = []
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(dev)
                out2d, _ = model_r(images, return_2d=True)
                features_2d.append(out2d.cpu().numpy())
                labels_list.append(labels.cpu().numpy())
        features_2d = np.concatenate(features_2d, axis=0)
        labels_list = np.concatenate(labels_list, axis=0)
        present_classes = np.unique(labels_list)
        n_classes = len(present_classes)
        custom_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        cmap = ListedColormap(custom_colors[:n_classes])
        plt.figure(figsize=(8, 8))
        limmm = 25
        xx, yy = np.meshgrid(np.linspace(-limmm, limmm, 300), np.linspace(-limmm, limmm, 300))
        grid = np.c_[xx.ravel(), yy.ravel()]
        grid_tensor = torch.tensor(grid, dtype=torch.float32).to(dev)
        with torch.no_grad():
            logits = model_r.linear(grid_tensor)
            Z = torch.argmax(logits, dim=1).cpu().numpy()
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, levels=np.arange(n_classes+1)-0.5, alpha=0.2, cmap=cmap)
        for i, cls in enumerate(present_classes):
            idxs = labels_list == cls
            # plt.scatter(features_2d[idxs, 0], features_2d[idxs, 1], label=f'Class {cls}', alpha=0.5, s=10, color=cmap(i))
        # Recompute features for original and perturbed points at this round
        with torch.no_grad():
            orig_feat2d, _ = model_r(orig_img, return_2d=True)
            orig_feat2d = orig_feat2d.cpu().numpy()[0]
            adv_feats2d = []
            for adv_img in adv_imgs:
                adv_feat2d, _ = model_r(adv_img, return_2d=True)
                adv_feats2d.append(adv_feat2d.cpu().numpy()[0])
            adv_feats2d = np.stack(adv_feats2d)
        plt.scatter(orig_feat2d[0], orig_feat2d[1], c='blue', marker='o', s=80, label='Original', alpha=0.4)
        for i in range(noae):
            plt.scatter(adv_feats2d[i, 0], adv_feats2d[i, 1], c='green', marker='o', s=50, alpha=0.4)
            plt.plot([orig_feat2d[0], adv_feats2d[i, 0]], [orig_feat2d[1], adv_feats2d[i, 1]], 'k--', linewidth=1, alpha=0.5)
        plt.legend()
        attack_name = attack_method.upper()
        if target_class == -1:
            plt.title(f'Decision Boundary at Round {r} (Untargeted {attack_name}s)')
        else:
            plt.title(f'Decision Boundary at Round {r} (Targeted {attack_name}s to class {target_class})')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        if target_class == -1:
            plt.savefig(f'ae624/round_{r}_{attack_method}{ori_cla}_untargeted.png')
        else:
            plt.savefig(f'ae624/round_{r}_{attack_method}{ori_cla}{target_class}.png')
        plt.close()