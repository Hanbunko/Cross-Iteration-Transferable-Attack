import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import random
from collections import defaultdict
from tqdm import tqdm

# Distance of different Classes in Feature Space

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

def extract_features(model, data_loader, device='cpu'):
    """Extract features from the penultimate layer (before the final classifier)"""
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(data_loader, desc="Extracting features")):
            data = data.to(device)
            # Get features from the model
            _, _, _, _, _, _, features_batch, _ = model(data, return_features=True)
            features.append(features_batch.cpu())
            labels.append(target)

    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    return features.numpy(), labels.numpy()


def calculate_class_statistics(features, labels, num_classes=10):
    """Calculate class centers and densities"""
    class_features = defaultdict(list)

    # Group features by class
    for feat, label in zip(features, labels):
        class_features[label].append(feat)

    # Convert to numpy arrays
    for label in class_features:
        class_features[label] = np.array(class_features[label])

    # Calculate class centers
    class_centers = {}
    for label in range(num_classes):
        if label in class_features:
            class_centers[label] = np.mean(class_features[label], axis=0)

    # Calculate densities (average distance from samples to their class center)
    class_densities = {}
    for label in range(num_classes):
        if label in class_features:
            distances = np.linalg.norm(class_features[label] - class_centers[label], axis=1)
            class_densities[label] = {
                'mean_distance': np.mean(distances),
                'std_distance': np.std(distances),
                'num_samples': len(class_features[label])
            }

    return class_centers, class_densities, class_features


def calculate_inter_class_distances(class_centers):
    """Calculate pairwise distances between class centers"""
    num_classes = len(class_centers)
    distance_matrix = np.zeros((num_classes, num_classes))

    for i in range(num_classes):
        for j in range(num_classes):
            if i != j:
                distance_matrix[i, j] = np.linalg.norm(class_centers[i] - class_centers[j])

    return distance_matrix


def main():
    # Set device
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    src_dic = "modsp"
    # Load model
    model = torch.load(f'{src_dic}/1000.pth')
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    # Load pretrained weights if available (optional)
    # You can train the model or load pretrained weights here
    # For this analysis, we'll use a randomly initialized model
    # In practice, you'd want to use a trained model for meaningful features

    # Prepare data
    _, _, test_loader, classes = prepare_cifar10(batch_size=128, device=device)

    # CIFAR-10 class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    # Extract features
    features, labels = extract_features(model, test_loader, device)

    # Calculate class statistics
    class_centers, class_densities, class_features = calculate_class_statistics(features, labels)

    # Print class densities
    print("\n=== Class Feature Densities ===")
    print("(Average distance from samples to their class center)")
    print("-" * 50)
    for i, class_name in enumerate(class_names):
        if i in class_densities:
            density = class_densities[i]
            print(f"{class_name:12s}: {density['mean_distance']:8.4f} ± {density['std_distance']:6.4f} "
                  f"(n={density['num_samples']})")

    # Calculate inter-class distances
    print("\n=== Inter-Class Distances ===")
    print("(Distance between class centers)")
    print("-" * 50)
    distance_matrix = calculate_inter_class_distances(class_centers)

    # Print distance matrix
    print("\nDistance Matrix:")
    print("       ", end="")
    for name in class_names:
        print(f"{name[:6]:>8s}", end="")
    print()

    for i, name1 in enumerate(class_names):
        print(f"{name1[:6]:6s}", end="")
        for j, name2 in enumerate(class_names):
            if i == j:
                print(f"{'---':>8s}", end="")
            else:
                print(f"{distance_matrix[i, j]:8.2f}", end="")
        print()

    # Find closest and farthest class pairs
    print("\n=== Class Pair Analysis ===")
    print("-" * 50)

    # Get upper triangle indices (to avoid duplicates)
    upper_indices = np.triu_indices(10, k=1)
    distances_list = []

    for i, j in zip(upper_indices[0], upper_indices[1]):
        distances_list.append((distance_matrix[i, j], class_names[i], class_names[j]))

    # Sort by distance
    distances_list.sort(key=lambda x: x[0])

    print("\nClosest class pairs:")
    for i in range(min(5, len(distances_list))):
        dist, class1, class2 = distances_list[i]
        print(f"{class1:12s} <-> {class2:12s}: {dist:8.4f}")

    print("\nFarthest class pairs:")
    for i in range(max(0, len(distances_list)-5), len(distances_list)):
        dist, class1, class2 = distances_list[i]
        print(f"{class1:12s} <-> {class2:12s}: {dist:8.4f}")

    # Additional analysis: Feature space statistics
    print("\n=== Overall Feature Space Statistics ===")
    print("-" * 50)

    all_features = np.vstack(list(class_features.values()))
    global_center = np.mean(all_features, axis=0)

    # Calculate average distance from global center for each class
    print("\nAverage distance from global center:")
    for i, class_name in enumerate(class_names):
        if i in class_centers:
            dist_to_global = np.linalg.norm(class_centers[i] - global_center)
            print(f"{class_name:12s}: {dist_to_global:8.4f}")

    # Feature dimensionality
    print(f"\nFeature dimensionality: {features.shape[1]}")
    print(f"Total samples analyzed: {features.shape[0]}")

    return class_centers, class_densities, distance_matrix


if __name__ == "__main__":
    class_centers, class_densities, distance_matrix = main()