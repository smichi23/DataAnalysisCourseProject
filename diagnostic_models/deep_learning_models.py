import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from torch.utils.data import Dataset


class DataWrapper(Dataset):
    """
    Class to wrap a dataset. Assumes X and y are already
    torch tensors and have the right data type and shape.

    Parameters
    ----------
    X : torch.Tensor
        Features tensor.
    y : torch.Tensor
        Labels tensor.
    """

    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class CNNClassifier2D(nn.Module):
    def __init__(self, input_size, **hp):
        super(CNNClassifier2D, self).__init__()
        self.input_size = input_size
        self.hp = hp
        self.set_delault_hp()
        self.build_model()

    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nbr_params = sum([np.prod(p.size()) for p in model_parameters])
        return f'Nbr of trainable parameters: {nbr_params}'

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nbr_params = sum([np.prod(p.size()) for p in model_parameters])
        return super(CNNClassifier2D, self).__str__() + f'\nNbr of trainable parameters: {nbr_params}'

    def set_delault_hp(self):
        default_hp = {"activation_func": nn.ReLU(), 'padding': '',
                      'kernel_initializer': 'glorot_normal', 'd_rate': 0.2,
                      'initial_kernel_size': (4, 4), 'final_activation_func': nn.Sigmoid(),
                      'kernel_size': (3, 3), "channels": 1,
                      "down_sampling_nb_layers": 4}
        for hpKey, hpValue in default_hp.items():
            if hpKey not in self.hp.keys():
                self.hp[hpKey] = hpValue

    def build_model(self):
        d_rate = self.hp['d_rate']
        for layers in range(self.hp["down_sampling_nb_layers"]):
            self.add_module(f'conv{layers}', nn.Conv2d(self.hp["channels"] * (2 ** layers),
                                                       self.hp["channels"] * (2 ** (layers + 1)),
                                                       self.hp['kernel_size'], 1))
            self.__getattr__(f'conv{layers}').weight.data.normal_(mean=0.0, std=1.0)
            self.add_module(f'conv{layers}_activation', self.hp["activation_func"])
            self.add_module(f'conv{layers}_dropout', nn.Dropout(d_rate))
            self.add_module(f'conv{layers}_pooling', nn.MaxPool2d(2))

        self.add_module(f'final_layer_lin', nn.Linear(self.hp["channels"] * (2 ** (self.hp["down_sampling_nb_layers"])), 1))
        self.add_module(f'final_layer', self.hp["final_activation_func"])

    def forward(self, x):
        for layers in range(self.hp["down_sampling_nb_layers"]):
            x = self.__getattr__(f'conv{layers}')(x)
            x = self.__getattr__(f'conv{layers}_activation')(x)
            x = self.__getattr__(f'conv{layers}_dropout')(x)
            x = self.__getattr__(f'conv{layers}_pooling')(x)

        x = self.__getattr__(f'final_layer_lin')(x.flatten(start_dim=-3))
        x = self.__getattr__(f'final_layer')(x)
        return x.flatten()

    def predict(self, x, threshold=0.5):
        output = self.forward(x)
        _, predicted = torch.greater(output.data, threshold)
        return predicted


class Trainer:
    def __init__(self, model, data_augmentation=False, **hp):
        self.model = model
        self.data_augmentation = data_augmentation
        self.hp = hp
        self.set_delault_hp()

    def set_delault_hp(self):
        default_hp = {'loss': nn.CrossEntropyLoss(), 'optimizer': Adam(self.model.parameters(), lr=0.000001),
                      'metrics': [self.accuracy], 'epochs': 100}
        for hpKey, hpValue in default_hp.items():
            if hpKey not in self.hp.keys():
                self.hp[hpKey] = hpValue

    def train(self, train_loader, val_loader):
        for epoch in range(self.hp['epochs']):
            train_loss = self._train_on_epoch(train_loader)
            print(f'Epoch: {epoch}, Train loss: {train_loss}')
            val_loss = self._validate(val_loader)
            print(f'Epoch: {epoch}, Validation loss: {val_loss}')

    def _train_on_epoch(self, train_loader):
        self.model.train()
        for features, labels in train_loader:
            self.hp['optimizer'].zero_grad()
            output = self.model(features)
            loss = self.hp['loss'](output, labels)
            loss.backward()
            self.hp['optimizer'].step()
        return loss.item()

    def _validate(self, val_loader):
        self.model.eval()
        with torch.no_grad():
            for features, labels in val_loader:
                out = self.model(features)
                loss = self.hp['loss'](out, labels)

        return loss.item()

    def predict(self, x, threshold=0.5):
        output = self.forward(x)
        _, predicted = torch.greater(output.data, threshold)
        return predicted

    def accuracy(self, output, target):
        _, predicted = torch.max(output.data, 1)
        correct = (predicted == target).sum().item()
        return correct / len(target)

    def get_model(self):
        return self.model
