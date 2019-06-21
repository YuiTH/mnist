#!python
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import torch.nn as nn
import torchvision
from torchvision import transforms
import torch.utils.data
import torch.optim
import numpy as np

from LeNet_torch import LeNet

root = './data'
PATH = './lenet_mnist_Adam'
trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (1.0,))
])

learning_rate = 1e-3  # hyperparameter
epoches = 50  # hyperparameter
val_every_epoche = 5

full_dataset = torchvision.datasets.MNIST(root=root, train=True, transform=trans, download=True)
dataset_size = len(full_dataset)
train_size = int(0.9 * dataset_size)  # hyperparameter

Val_size = dataset_size - train_size
train_set, val_set = torch.utils.data.random_split(full_dataset, [train_size, Val_size])

data_loader_train = torch.utils.data.DataLoader(dataset=train_set, batch_size=64, shuffle=True)
data_loader_val = torch.utils.data.DataLoader(dataset=val_set, batch_size=64, shuffle=True)

lenet = LeNet().cuda()
optim = torch.optim.Adam(lenet.parameters())
# optim = torch.optim.SGD(lenet.parameters(), lr=learning_rate, momentum=0.9)
criterion = nn.CrossEntropyLoss()
loss_list = []
val_list = []


def validation():
    acc = 0
    with torch.no_grad():
        correct = 0
        total = 0
        #     for image, labels in data_loader_test:
        for image, labels in data_loader_val:
            image, labels = image.cuda(), labels.cuda()
            outputs = lenet(image)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            acc = correct / total
            print("validation acc", acc)

        print('Accuracy of the network on the validation test images: {} %'.format(100 * correct / total))

    return acc


def batch_test(image, labels):
    with torch.no_grad():
        correct = 0
        total = 0
        #     for image, labels in data_loader_test:
        image, labels = image.cuda(), labels.cuda()
        outputs = lenet(image)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        acc = correct / total
        print("acc", acc)
    return acc


def train():
    acc = 0
    for epoche in range(epoches):
        print("epoche:", epoche, " start")
        image, label = None, None
        for (image, label) in data_loader_train:
            # image = torchvision.transforms.functional.resize(image)
            res = lenet(image.cuda())
            loss = criterion(res, label.cuda())
            print("epoche :", epoche, " current acc=", acc, " custom loss=", loss.item())
            loss_list.append(loss.item())
            optim.zero_grad()
            loss.backward()
            optim.step()
        acc = batch_test(image, label)
        if epoche % 5 == 0:
            torch.save({'epoch': epoche + 1, 'state_dict': lenet.state_dict(),
                        'optimizer': optim.state_dict()},
                       PATH + str(epoche))
        loss_np = np.array(loss_list)
        np.save("loss.npy", loss_np)
        print(loss_np.shape)
        if epoche % val_every_epoche == 0:
            val_list.append(validation())
            np.save("val.npy", np.array(val_list))


train()
