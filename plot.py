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

# this file is for plotting the images for {original image, perturbed image}

id = 4

loc = ['airplane', 'automobile', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck']

dev = torch.device('cuda')

def imshow(img, title = "default"):
    # denormalize (approx)
    # img = img * 0.2 + 0.48

    npimg = img.cpu().detach().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.savefig(f'./nimg/{title}.png')
    plt.show()

def rescale(img):
    cp = img.clone()
    return (img - img.min()) / (img.max() - img.min() + 1e-8)

tra_tran, tes_tran = [], []
siz = 4
tra_tran += [transforms.RandomCrop(2 ** siz, padding=siz), transforms.RandomHorizontalFlip()]
tra_tran += [transforms.ToTensor()]
tes_tran += [transforms.ToTensor()]
tra_set = datasets.CIFAR10('./data', download=True, transform=transforms.Compose(tra_tran))
tes_set = datasets.CIFAR10('./data', download=True, transform=transforms.Compose(tes_tran))

train_loader = DataLoader(tra_set, batch_size=64, shuffle=True)
test_loader = DataLoader(tes_set, batch_size=64, shuffle=False)

# Pick a random index from class 0 (airplane) in the test set
class0_indices = [i for i, t in enumerate(tes_set.targets) if t == 0]
random_id = random.choice(class0_indices)

imgs, lbls = next(iter(test_loader))
# Find the batch that contains the random_id
batch_idx = random_id // test_loader.batch_size
in_batch_idx = random_id % test_loader.batch_size
for i, (batch_imgs, batch_lbls) in enumerate(test_loader):
    if i == batch_idx:
        imgs = batch_imgs
        lbls = batch_lbls
        break

imshow(torchvision.utils.make_grid(imgs[in_batch_idx]))

model = models.resnet18()
model.conv1 = nn.Conv2d(3,64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
model.maxpool = nn.Identity()
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(dev)
model = torch.load(f"./modsp/900.pth")

model.eval()

correct_clean = 0
total = 0
for images, labels in test_loader:
    images, labels = images.to(dev), labels.to(dev)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct_clean += (predicted == labels).sum().item()
print('accuracy: %.2f %%' % (100 * correct_clean / total))

with torch.no_grad():
    outputs = model(imgs.to(dev))
    probs = F.softmax(outputs, dim=1)
    ori_conf, ori_pred = torch.max(probs, dim=1)
    print(ori_pred[in_batch_idx])
    cp = ori_conf[in_batch_idx]*100
    print(f"cp:{cp:.2f}%")
imshow(torchvision.utils.make_grid(imgs[in_batch_idx]), title=f"{loc[ori_pred[in_batch_idx].item()]} with confidence {cp:.2f}%")

def pgd_attack(model, images, labels, eps=0.031, alpha=0.008, iters=80):
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


def targeted_pgd_attack(model, images, target_labels, eps=0.031, alpha=0.007, iters=200):
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

model.eval()

ai, ptb = pgd_attack(model, imgs, lbls)
with torch.no_grad():
    adout = model(ai.to(dev))
    adpro = F.softmax(adout, dim=1)
    adv_conf, adv_pred = adpro.max(dim=1)
    print(adv_pred[in_batch_idx])
    ap = adv_conf[in_batch_idx] * 100
    print(f"ap:{ap:.2f}%")
imshow(torchvision.utils.make_grid(rescale(ptb[in_batch_idx])), title = "untargeted perturbation (rescaled)")
imshow(torchvision.utils.make_grid(ai[in_batch_idx]), title = f"untargeted: {loc[adv_pred[in_batch_idx].item()]} with confidence {ap:.2f}%")

for i in range(10):
    target_labels = torch.full_like(lbls, i).to(dev)
    ai, ptb = targeted_pgd_attack(model, imgs, target_labels)
    with torch.no_grad():
        adout = model(ai.to(dev))
        adpro = F.softmax(adout, dim=1)
        adv_conf, adv_pred = adpro.max(dim=1)
        print(adv_pred[in_batch_idx])
        ap = adv_conf[in_batch_idx] * 100
        print(f"ap:{ap:.2f}%")
    imshow(torchvision.utils.make_grid(rescale(ptb[in_batch_idx])), title=f"targeted at {i}({loc[i]}) perturbation (rescaled)")
    imshow(torchvision.utils.make_grid(ai[in_batch_idx]), title = f"{loc[adv_pred[in_batch_idx].item()]} with confidence {ap:.2f}%")

