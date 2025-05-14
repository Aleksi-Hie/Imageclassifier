
from torchvision.io import decode_image
from torchvision.models import mobilenet_v3_small
from torchvision.datasets import MNIST
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


from env import path

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Convert 1 channel to 3 channels
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize for 3 channels
])

def showimages(img, one_channel = False):
    if one_channel:
        img = img.mean(dim=0)
    img = img/2 +0.5
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmpa="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1,2,0)))

training_data = MNIST(path, train=True, transform=transform, download = True)
training_loader = torch.utils.data.DataLoader(training_data, batch_size=4, shuffle=True, num_workers=2)
model = mobilenet_v3_small()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum=0.9)
loss_fn = torch.nn.CrossEntropyLoss()

def main():
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    print("Training set has {} instances".format(len(training_data)))

    
    dataiter = iter(training_loader)
    images, labels = dataiter.next()
    img_grid = torchvision.utils.make_grid(images)
    showimages(img_grid, one_channel = True)
    print(' '.join(classes[labels[j]] for j in range(4)))


def train_epoch(idx, tb_writer):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(training_loader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()

        optimizer.step()
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss /1000
            print("batch {} loss: {}".format(i+1, last_loss))
            tb_x = idx * len(training_loader)+i+1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0
    return last_loss

EPOCHS = 10
best_vloss = 10000000
timestamp = datetime.now().strftime("%d/%m/%y_%H%M%S")
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
for e in range(EPOCHS):
    print('EPOCH {}'.format(e+1))
    model.train(True)
    avg_loss = train_epoch(e, writer)
    running_vloss = 0.0
    model.eval()
    epoch_number += 1