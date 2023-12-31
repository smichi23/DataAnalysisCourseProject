import datetime
import os
from copy import deepcopy

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from torch.utils.data import Dataset, DataLoader


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

    def __init__(self, X, y, transforms=None, device='cpu'):
        tmp_features = []
        for features in X:
            tmp_features.append(features - features.min() / (features.max() - features.min()))
        self.labels = torch.as_tensor(y, dtype=torch.float32).data.to(device)
        self.features = torch.as_tensor(np.asarray(tmp_features), dtype=torch.float32).unsqueeze(1).data.to(device)
        self.transforms = transforms

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.transforms is not None:
            return self.transforms(self.features[idx]), self.labels[idx]
        else:
            return self.features[idx], self.labels[idx]


class CNNClassifier2D(nn.Module):
    def __init__(self, input_size, **hp):
        super(CNNClassifier2D, self).__init__()
        self.input_size = input_size
        self.hp = hp
        self.set_delault_hp()
        self.build_model()
        self.initial_weights = deepcopy(self.state_dict())
        self.trainer = None
        self.name = 'CNNClassifier2D'
        self.threshold_domain = [0, 1]

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
                      "down_sampling_nb_layers": 4, "transform": None, "device": "cpu"}
        for hpKey, hpValue in default_hp.items():
            if hpKey not in self.hp.keys():
                self.hp[hpKey] = hpValue

    def build_model(self):
        d_rate = self.hp['d_rate']
        self.add_module(f'conv_initial',
                        nn.Conv2d(self.hp["channels"], self.hp["channels"], self.hp['initial_kernel_size'], 1))
        for layers in range(self.hp["down_sampling_nb_layers"]):
            self.add_module(f'conv{layers}', nn.Conv2d(self.hp["channels"] * (2 ** layers),
                                                       self.hp["channels"] * (2 ** (layers + 1)),
                                                       self.hp['kernel_size'], 1))
            self.__getattr__(f'conv{layers}').weight.data.normal_(mean=0.0, std=1.0)
            self.add_module(f'conv{layers}_activation', self.hp["activation_func"])
            self.add_module(f'conv{layers}_dropout', nn.Dropout(d_rate))
            self.add_module(f'conv{layers}_pooling', nn.MaxPool2d(2))

        self.add_module(f'final_layer_lin',
                        nn.Linear(self.hp["channels"] * (2 ** (self.hp["down_sampling_nb_layers"])), 1))
        self.add_module(f'final_layer', self.hp["final_activation_func"])

    def forward(self, x):
        x = self.__getattr__(f'conv_initial')(x)
        for layers in range(self.hp["down_sampling_nb_layers"]):
            x = self.__getattr__(f'conv{layers}')(x)
            x = self.__getattr__(f'conv{layers}_activation')(x)
            x = self.__getattr__(f'conv{layers}_dropout')(x)
            x = self.__getattr__(f'conv{layers}_pooling')(x)

        x = self.__getattr__(f'final_layer_lin')(x.flatten(start_dim=-3))
        x = self.__getattr__(f'final_layer')(x)
        return x.flatten()

    def predict(self, x, threshold=0.5, training=False):
        if type(x) != torch.Tensor:
            tmp_features = []
            for features in x:
                tmp_features.append(features - features.min() / (features.max() - features.min()))
            data = torch.as_tensor(np.asarray(tmp_features), dtype=torch.float32).unsqueeze(1).to(self.hp["device"])
        else:
            data = x.to(self.hp["device"])
        output = self.forward(data)
        predicted = torch.greater(output.data, threshold)
        if training:
            return predicted
        else:
            return predicted.cpu()

    def reset(self):
        self.load_state_dict(self.initial_weights)

    def fit(self, data, labels, batch_size=20, shuffle=True):
        assert self.trainer is not None, 'Trainer not set'
        self.reset()
        ml_data_wrapped = DataWrapper(data, labels, transforms=self.hp["transform"], device=self.hp["device"])
        data_loader = DataLoader(ml_data_wrapped, batch_size=batch_size, shuffle=shuffle)
        self.trainer.train(data_loader, data_loader)

    def set_trainer(self, trainer):
        self.trainer = trainer


class Trainer:
    def __init__(self, model, **hp):
        self.model = model
        self.hp = hp
        self.training_loss = []
        self.validation_loss = []
        self.metrics_training = {}
        self.metrics_validation = {}
        self.set_delault_hp()
        self.initial_lr = self.hp['optimizer'].param_groups[0]['lr']
        if self.hp["lr_scheduler"] is not None:
            self.initial_scheduler_state = self.hp["lr_scheduler"].state_dict()

    def set_delault_hp(self):
        default_hp = {'loss': nn.CrossEntropyLoss(), 'optimizer': Adam(self.model.parameters(), lr=0.000001),
                      'metrics': [self.accuracy], 'epochs': 100, "show_plots_every_training": False,
                      "early_stoping_condition": None, "lr_scheduler": None, "save_path": os.path.dirname(__file__)}
        for hpKey, hpValue in default_hp.items():
            if hpKey not in self.hp.keys():
                self.hp[hpKey] = hpValue

    def train(self, train_loader, val_loader):
        self.reset()
        for epoch in tqdm(range(self.hp['epochs'])):
            train_loss = self._train_on_epoch(train_loader)
            self.training_loss.append(train_loss)
            for metric in self.hp['metrics']:
                if metric.__name__ not in self.metrics_training.keys():
                    self.metrics_training[metric.__name__] = []
                self.metrics_training[metric.__name__].append(torch.mean(metric(train_loader)))
            val_loss = self._validate(val_loader)
            self.validation_loss.append(val_loss)
            for metric in self.hp['metrics']:
                if metric.__name__ not in self.metrics_validation.keys():
                    self.metrics_validation[metric.__name__] = []
                self.metrics_validation[metric.__name__].append(torch.mean(metric(val_loader)))

            if self.hp["lr_scheduler"] is not None:
                self.hp["lr_scheduler"].step()
            if self.hp["early_stoping_condition"] is not None:
                if self.hp["early_stoping_condition"][1] == 'metric':
                    if self.hp["early_stoping_condition"][0](
                            self.metrics_validation[self.hp["early_stoping_condition"][2]]):
                        break

                if self.hp["early_stoping_condition"][1] == 'loss':
                    if self.hp["early_stoping_condition"][0](self.validation_loss):
                        break
        if self.hp["show_plots_every_training"]:
            self.plot_training_and_validation_loss_and_metrics()

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

    def accuracy(self, data_loader):
        score = []
        for features, labels in data_loader:
            predicted = self.model.predict(features, training=True)
            correct = torch.eq(predicted, labels).sum()
            score.append(correct / len(labels))
        return torch.as_tensor(score, dtype=torch.float32)

    def plot_training_and_validation_loss_and_metrics(self):
        nb_metrics = len(self.hp['metrics'])
        fig, ax = plt.subplots(nb_metrics + 1, 1, figsize=(7.5, 10), dpi=100)
        epochs = list(range(len(self.training_loss)))

        ax[0].plot(epochs, self.training_loss, label='Training loss')
        ax[0].plot(epochs, self.validation_loss, label='Validation loss')
        for metric in self.hp['metrics']:
            ax[1].plot(epochs, self.metrics_training[metric.__name__], label=f'Training {metric.__name__}')
            ax[1].plot(epochs, self.metrics_validation[metric.__name__], label=f'Validation {metric.__name__}')
            ax[1].set_ylabel(metric.__name__ + " [-]")
        ax[1].legend()
        ax[0].legend()
        ax[0].set_ylabel('Loss [-]')
        ax[-1].set_xlabel('Epochs [-]')
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        plt.savefig(os.path.join(self.hp["save_path"], f"training_{current_time}.png"), bbox_inches='tight')
        plt.close()

    def get_model(self):
        return self.model

    def reset(self):
        self.training_loss = []
        self.validation_loss = []
        self.metrics_training = {}
        self.metrics_validation = {}
        self.model.load_state_dict(self.model.initial_weights)
        if self.hp["lr_scheduler"] is not None:
            self.hp["lr_scheduler"].load_state_dict(self.initial_scheduler_state)
        self.hp['optimizer'].param_groups[0]['lr'] = self.initial_lr
