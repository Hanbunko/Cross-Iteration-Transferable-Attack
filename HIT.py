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
import cv2
from PIL import Image

def set_seed(seed):
    # Ensure reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def best_device():
    if torch.cuda.is_available():
        return 'cuda'
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


# ====================
# PGD White-box Attack
# ====================
def pgd_attack(model, images, labels, epsilon=8/255., alpha=2/255., iters=20):
    images = images.clone().detach().to(images.device)
    ori_images = images.clone().detach()
    delta = torch.zeros_like(images).uniform_(-epsilon, epsilon).to(images.device)
    delta.requires_grad = True

    for _ in range(iters):
        outputs = model(images + delta)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()

        delta.data = delta + alpha * delta.grad.sign()
        delta.data = torch.clamp(delta.data, -epsilon, epsilon)
        delta.data = torch.clamp(images + delta.data, 0, 1) - images
        delta.grad.zero_()

    return torch.clamp(images + delta.data, 0, 1)

# ============================
# No-box HIT (frequency-based)
# ============================

def hit_no_box_attack(images, epsilon=8/255., cutoff=0.25, enhance_factor=1.8):
    """
    HIT (Hybrid Image Transformation) no-box attack for CIFAR-10
    Optimized for higher ASR on 32x32 images

    Args:
        images: Input images tensor (B, 3, 32, 32)
        epsilon: Maximum perturbation budget (default 8/255 for CIFAR-10)
        cutoff: Frequency cutoff parameter (0-1, where lower means more low-freq)
        enhance_factor: Weight factor for high-frequency components

    Returns:
        Adversarial images with HIT perturbations
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    device = images.device
    batch_size = images.shape[0]

    # Step 1: Generate HIGH-FREQUENCY pattern for CIFAR-10
    pattern_size = 32

    # Create a high-frequency checkerboard-like pattern mixed with circles
    pattern = np.zeros((pattern_size, pattern_size, 3), dtype=np.float32)

    # Add checkerboard base pattern for high frequency
    checker_size = 2  # Very small squares for high frequency
    for i in range(0, pattern_size, checker_size):
        for j in range(0, pattern_size, checker_size):
            if (i // checker_size + j // checker_size) % 2 == 0:
                pattern[i:i+checker_size, j:j+checker_size] = [1, 0, 0]  # Red
            else:
                pattern[i:i+checker_size, j:j+checker_size] = [0, 0, 1]  # Blue

    # Overlay with concentric circles for additional frequency components
    center = pattern_size // 2
    Y, X = np.ogrid[:pattern_size, :pattern_size]
    dist_from_center = np.sqrt((X - center)**2 + (Y - center)**2)

    # Add multiple frequency components
    for freq in [4, 8, 12]:
        mask = np.sin(dist_from_center * freq * np.pi / pattern_size) > 0
        pattern[mask] = 1 - pattern[mask]  # Invert colors for frequency mixing

    # Normalize pattern to [0, 1]
    pattern = np.clip(pattern, 0, 1)

    # Save pattern
    pattern_pil = Image.fromarray((pattern * 255).astype(np.uint8))
    pattern_pil.save('tile_pattern_cifar.png')

    # Load pattern as tensor
    pattern_tensor = transforms.ToTensor()(pattern_pil).unsqueeze(0).to(device)
    pattern_batch = pattern_tensor.expand(batch_size, 3, 32, 32)

    # Step 2: Create VERY SMALL Gaussian kernel for CIFAR-10
    # This is CRITICAL - we need much smaller kernel for 32x32 images
    # Using cutoff_frequency = 1 gives us a 3x3 kernel
    cutoff_frequency = 1
    s = cutoff_frequency * 0.5  # Even smaller sigma for sharper frequency separation
    k = 1  # Results in 3x3 kernel

    # Create Gaussian kernel
    coords = np.arange(-k, k+1, dtype=np.float32)
    probs = np.exp(-coords**2 / (2*s*s)) / np.sqrt(2*np.pi*s*s)
    kernel = np.outer(probs, probs)

    # Normalize kernel
    kernel = kernel / kernel.sum()

    # Convert to torch tensor
    gaussian_kernel = torch.from_numpy(kernel).float().to(device)
    gaussian_kernel = gaussian_kernel.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)

    # Step 3: Apply frequency separation
    padding = k

    # Extract low frequencies from original image
    low_freq_images = F.conv2d(images, gaussian_kernel, padding=padding, groups=3)

    # Extract high frequencies from pattern
    low_freq_pattern = F.conv2d(pattern_batch, gaussian_kernel, padding=padding, groups=3)
    high_freq_pattern = pattern_batch - low_freq_pattern

    # Step 4: Create hybrid with STRONG high-frequency injection
    # Key insight: CIFAR-10 needs much stronger HF perturbations
    weight_factor = 5.0  # Much higher for CIFAR-10

    # Alternative mixing strategy for better attack
    # Instead of just adding, we also suppress some original high frequencies
    high_freq_images = images - low_freq_images

    # Hybrid image: Keep low freq from original, replace high freq with pattern
    hybrid_images = low_freq_images + weight_factor * high_freq_pattern - 0.5 * high_freq_images

    # Step 5: Maximize perturbation usage within epsilon ball
    perturbation = hybrid_images - images

    # Normalize perturbation to use full epsilon budget
    # Use L-infinity norm
    perturbation_flat = perturbation.view(batch_size, -1)
    perturbation_max = perturbation_flat.abs().max(dim=1, keepdim=True)[0]
    perturbation_max = perturbation_max.view(batch_size, 1, 1, 1)

    # Scale to use full epsilon
    scale = epsilon / (perturbation_max + 1e-10)
    scale = torch.min(scale, torch.ones_like(scale))
    perturbation = perturbation * scale

    # Apply random sign to increase diversity
    random_sign = torch.sign(torch.randn_like(perturbation))
    perturbation = perturbation * random_sign

    # Ensure within bounds
    perturbation = torch.clamp(perturbation, -epsilon, epsilon)

    # Generate final adversarial examples
    adv_images = torch.clamp(images + perturbation, 0, 1)

    # Clean up
    if os.path.exists('tile_pattern_cifar.png'):
        os.remove('tile_pattern_cifar.png')

    return adv_images

# ==========================
# Main Evaluation
# ==========================
def evaluate_asr(model, adv_images, target_labels):
    model.eval()
    with torch.no_grad():
        preds = model(adv_images).argmax(1)
        return (preds != target_labels).float().mean().item() * 100

def main():
    ckpt_dir = 'modsp'; set_seed(2025)
    device = best_device()
    model, _, test_loader, classes = prepare_cifar10(batch_size=128, device=device)

    if not os.path.exists(os.path.join(ckpt_dir, '100.pth')):
        print(f"Error: Model file 'modsp/100.pth' not found.")
        print("Please ensure you have the pre-trained models from the FL simulation in a 'modsp' directory.")
        exit()

    model_100 = torch.load(os.path.join(ckpt_dir, '100.pth'))
    model_100.to(device)
    model_100.eval()
    criterion = nn.CrossEntropyLoss()

    # Evaluate on one batch
    data_iter = iter(test_loader)
    data, target = next(data_iter)
    data, target = data.to(device), target.to(device)

    print("="*60)
    ae_pgd = pgd_attack(model_100, data, target, epsilon=8/255., alpha=2/255., iters=40)
    ae_hit = hit_no_box_attack(data, epsilon=25.5/255., cutoff=0.25, enhance_factor=1.8)

    rounds = list(range(100, 1100, 100))
    pgd_asrs, hit_asrs = [], []
    for r in rounds:
        model = torch.load(f'modsp/{r}.pth')
        model.to(device)
        model.eval()

        with torch.no_grad():
            asr_pgd = evaluate_asr(model, ae_pgd, target)
            asr_hit = evaluate_asr(model, ae_hit, target)
            pgd_asrs.append(asr_pgd)
            hit_asrs.append(asr_hit)
    print(f"\nPGD ASR List: {[round(asr, 2) for asr in pgd_asrs]}")
    print(f"HIT ASR List: {[round(asr, 2) for asr in hit_asrs]}")
    print("="*60)


if __name__ == '__main__':
    main()