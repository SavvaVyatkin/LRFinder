import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import lightning.pytorch as pl

from callback import LRFinder

# Define a simple ResNet model for MNIST
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# PyTorch Lightning Module
class ResNetMNIST(pl.LightningModule):
    def __init__(self, learning_rate=0.001):
        super(ResNetMNIST, self).__init__()
        self.train_outputs = {"loss": [], "preds": [], "targets": []}
        self.validation_outputs = {"loss": [], "preds": [], "targets": []}
        self.test_outputs = {"loss": [], "preds": [], "targets": []}
        self.learning_rate = learning_rate
        self.model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)

    def forward(self, x):
        return self.model(x)
    
    def shared_step(self, batch, batch_idx):
        spectra, label = batch
        output = self.model(spectra)
        loss = F.cross_entropy(output, label)

        return {"loss": loss, "preds": output, "targets": label}
    
    def training_step(self, batch, batch_idx):
        output = self.shared_step(batch, batch_idx)
        # print(output['loss'].detach())
        self.log(
            "metrics/train/loss", output["loss"].detach(), on_step=True, on_epoch=True
        )

        self.train_outputs["loss"].append(output["loss"].detach().cpu().numpy())
        self.train_outputs["preds"].append(output["preds"].detach().cpu().numpy())
        self.train_outputs["targets"].append(output["targets"].detach().cpu().numpy())

        return output
    
    def validation_step(self, batch, batch_idx):
        output = self.shared_step(batch, batch_idx)
        self.validation_outputs["loss"].append(output["loss"].detach().cpu().numpy())
        self.validation_outputs["preds"].append(output["preds"].detach().cpu().numpy())
        self.validation_outputs["targets"].append(
            output["targets"].detach().cpu().numpy()
        )

        return output
    

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

# Prepare DataLoader
def get_data_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Initialize the model with an initial learning rate
model = ResNetMNIST(learning_rate=0.001)

def run_lr_finder(*, weight_decay: float, dropout: float, num_training: int=1500, val_check_interval: float = 0.2, min_lr: float=1e-6, max_lr: float=1e-2, RANDOM_SEED: int = 4570):

    train_loader, val_loader = get_data_loaders()

    model = ResNetMNIST(learning_rate=0.001)


    # # # Run learning rate finder
    lr_finder = LRFinder(min_lr=min_lr, max_lr=max_lr)

    trainer = pl.Trainer(
        max_steps=num_training,
        logger=None,
        val_check_interval=val_check_interval,
        limit_val_batches=150,
        accelerator="gpu",
        callbacks=[lr_finder],
        gradient_clip_val=1.0,
        # enable_progress_bar=False,
    )

    trainer.fit(
        model, train_dataloaders=train_loader, val_dataloaders=val_loader
    )

    # Plot with
    plt.plot(np.log10(lr_finder.train_lrs), lr_finder.train_loss)
    plt.plot(np.log10(lr_finder.val_lrs), lr_finder.val_loss)
    plt.title(f'{weight_decay=}; {dropout=}')
    plt.show()


    return lr_finder

def run_multi_test(wd_tests: list, *, dropout: float, num_training: int = 1500, min_lr: float=1e-6, max_lr: float=1e-2, RANDOM_SEED: int = 24156) -> dict:
    results = {}

    for wd in wd_tests:
        lr_finder = run_lr_finder(weight_decay=wd, dropout=dropout, num_training=num_training, min_lr=min_lr, max_lr=max_lr, RANDOM_SEED=RANDOM_SEED)

        results[f'wd_{wd}'] = lr_finder

    return results
wd_tests = [0.1, 0.01]
lr_results = run_multi_test(wd_tests, dropout=0.5, num_training=5000, max_lr=3e-2)