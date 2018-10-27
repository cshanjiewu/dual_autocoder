import torch
import torch.cuda as cuda
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.autograd import Variable
from ImageNet import ImageNet
# Torchvision module contains various utilities, classes, models and datasets
# used towards computer vision usecases
from torchvision import datasets
from torchvision import transforms
from torchvision import models
from torch.nn import functional as F

#pathlib can handle the filePath across platform
from pathlib import Path, PureWindowsPath
# transformation defined the function of data argument
# ToTensor() :   Converts a PIL Image or numpy.ndarray (H x W x C) in the range
#     [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].


def show_img(orig, noisy, denoised):
    fig = plt.figure()

    orig = orig.swapaxes(0, 1).swapaxes(1, 2)
    noisy = noisy.swapaxes(0, 1).swapaxes(1, 2)
    denoised = denoised.swapaxes(0, 1).swapaxes(1, 2)

    # Normalize for display purpose
    orig = (orig - orig.min()) / (orig.max() - orig.min())
    noisy = (noisy - noisy.min()) / (noisy.max() - noisy.min())
    denoised = (denoised - denoised.min()) / (denoised.max() - denoised.min())

    fig.add_subplot(1, 3, 1, title='Original')
    plt.imshow(orig)

    fig.add_subplot(1, 3, 2, title='Noisy')
    plt.imshow(noisy)

    fig.add_subplot(1, 3, 3, title='Denoised')
    plt.imshow(denoised)

    fig.subplots_adjust(wspace=0.5)
    plt.show()


def train():

    # model = models.inception_v3(pretrained=True)

    # hyper-parameter
    batch_size = 1  # Reduce this if you get out-of-memory error
    learning_rate = 0.001
    noise_level = 0.1
    epoch = 10
    weight_re = 0.00001
    weight_cls = 1000

    data_dir = Path('D:/experiment_data/test')

    # Data loading code
    traindir = data_dir / 'train'
    traindir = str(traindir)

    valdir = data_dir / 'val'
    valdir = str(valdir)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transformations_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
        ])

    train_dataset = ImageNet(traindir, transformations_train)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=1)

    transformations_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    val_loader = ImageNet(valdir, transformations_val)

    val_loader = torch.utils.data.DataLoader(
        ImageNet(valdir, transformations_val),
        batch_size=batch_size, shuffle=True,
        num_workers=1)

    # Attention: Must before the construction of optimizer

    autoencoder = DenoisingAutoencoder().cuda()  # Moves all model parameters and buffers to the GPU.
    classifier  = Classifier().cuda()
    attacked = Attacked().cuda()

    parameters = list(autoencoder.parameters())

    # the first loss for reconstruction, the second for classifier to add noise
    loss_reconstruction = nn.MSELoss()
    loss_classifier = nn.NLLLoss()

    # construct a optimizer
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)

    # train
    train_loss = []
    valid_loss = []

    for i in range(epoch):

        # Let's train the model
        total_loss = 0.0
        total_iter = 0
        print("Iteration ", i + 1)
        autoencoder.train()  # set mode to train

        for image in train_loader:

            # image: batch size(250) * 3 * 32 * 32
            # label: 250
            image_n = image
            # move the datasets to GPU
            image = Variable(image).cuda()
            image_n = Variable(image_n).cuda()
            # Clears the gradients of all optimized torch.Tensors
            optimizer.zero_grad()
            # return the reconstructed image
            image_re = autoencoder(image_n)

            # return the result of softmax function
            _, predicted = torch.max(classifier(image_re), dim=1)

            attacked_predicted = torch.max(attacked(image_re), dim=1)
            attacked_max_pro, _ = torch.max(attacked_predicted)
            _, attacked_second = torch.max(attacked_predicted[attacked_predicted < attacked_max_pro])

            ground_label = attacked_second
            # loss is a 0-dim vector
            loss_re = loss_reconstruction(image_re, image)
            loss_predic = loss_classifier(predicted, ground_label)

            loss = weight_re * loss_re + weight_cls * loss_predic
            # calculate the gradients
            loss.backward()
            # update the parameters
            optimizer.step()

            total_iter += 1
            total_loss += loss.data.item()

        # Let's record the validation loss

        total_val_loss = 0.0
        total_val_iter = 0
        autoencoder.eval()
        for image, label in val_loader:
            noise = torch.randn(image.shape[0], 3, 32, 32) * noise_level
            image_n = torch.add(image, noise)

            image = Variable(image).cuda()
            image_n = Variable(image_n).cuda()

            output = autoencoder(image_n)
            loss = loss_reconstruction(output, image)

            total_val_iter += 1
            total_val_loss += loss.data.item()

        # Let's visualize the first image of the last batch in our validation set
        orig = image[0].cpu()
        noisy = image_n[0].cpu()
        denoised = output[0].cpu()

        orig = orig.data.numpy()
        noisy = noisy.data.numpy()
        denoised = denoised.data.numpy()

        show_img(orig, noisy, denoised)

        train_loss.append(total_loss / total_iter)
        valid_loss.append(total_val_loss / total_val_iter)

    # Save the model
    torch.save(autoencoder.state_dict(), "./5.autoencoder.pth")

    fig = plt.figure(figsize=(10, 7))
    plt.plot(train_loss, label='Train loss')
    plt.plot(valid_loss, label='Validation loss')
    plt.legend()
    plt.show()

    import random

    img, _ = random.choice(val_loader)
    img    = img.resize_((1, 3, 32, 32))
    noise  = torch.randn((1, 3, 32, 32)) * noise_level
    img_n  = torch.add(img, noise)

    img_n = Variable(img_n).cuda()
    denoised = autoencoder(img_n)
    show_img(img[0].numpy(), img_n[0].data.cpu().numpy(), denoised[0].data.cpu().numpy())


class DenoisingAutoencoder(nn.Module):

    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        # 32 x 32 x 3 (input)

        self.conv1e = nn.Conv2d(3, 24, 3, padding=2)  # 30 x 30 x 24
        self.conv2e = nn.Conv2d(24, 48, 3, padding=2)  # 28 x 28 x 48
        self.conv3e = nn.Conv2d(48, 96, 3, padding=2)  # 26 x 26 x 96
        self.conv4e = nn.Conv2d(96, 128, 3, padding=2)  # 24 x 24 x 128
        self.conv5e = nn.Conv2d(128, 256, 3, padding=2)  # 22 x 22 x 256
        self.mp1e = nn.MaxPool2d(2, return_indices=True)  # 11 x 11 x 256

        self.mp1d = nn.MaxUnpool2d(2)                   # 22 x 22 x 256
        self.conv5d = nn.ConvTranspose2d(256, 128, 3, padding=2)# 24 x 24 x 128
        self.conv4d = nn.ConvTranspose2d(128, 96, 3, padding=2)# 26 x 26 x 96
        self.conv3d = nn.ConvTranspose2d(96, 48, 3, padding=2)# 28 x 28 x 48
        self.conv2d = nn.ConvTranspose2d(48, 24, 3, padding=2) # 30 x 30 x 24
        self.conv1d = nn.ConvTranspose2d(24, 3, 3, padding=2)#32 x 32 x 3

    def forward(self, x):
        # Encoder
        x = self.conv1e(x)
        x = F.relu(x)
        x = self.conv2e(x)
        x = F.relu(x)
        x = self.conv3e(x)
        x = F.relu(x)
        x = self.conv4e(x)
        x = F.relu(x)
        x = self.conv5e(x)
        x = F.relu(x)
        x, i = self.mp1e(x)

        # Decoder
        x = self.mp1d(x, i)
        x = self.conv5d(x)
        x = F.relu(x)
        x = self.conv4d(x)
        x = F.relu(x)
        x = self.conv3d(x)
        x = F.relu(x)
        x = self.conv2d(x)
        x = F.relu(x)
        x = self.conv1d(x)
        x = F.relu(x)
        return x


class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()
        load_model = models.alexnet()
        load_model.load_state_dict(torch.load('./net/alexnet.pth'))
        self.classifier = load_model
        self.softmaxFunc = nn.Softmax()

    def forward(self, x):
        x = self.classifier(x)
        x = self.softmaxFunc(x)
        return x


class Attacked(nn.Module):

    def __init__(self):
        super(Attacked, self).__init__()
        load_model = models.inception_v3()
        load_model.load_state_dict(torch.load('./net/inception_v3_google.pth'))
        self.classifier = load_model
        self.softmaxFunc = nn.Softmax()

    def forward(self, x):
        x = self.classifier(x)
        x = self.softmaxFunc(x)
        return x


if __name__ == '__main__':
    train()
