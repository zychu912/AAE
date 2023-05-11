#!/usr/bin/env python
"""
Author: Zhiyuan Chu
Date: March 27, 2023
File Name: layer.py
"""

import torch
import torch.nn as nn

def build_network(layers: list[int], activation: nn.Module = nn.ReLU(),
                  batch_norm: bool = False, dropout: float = 0.0) -> nn.Sequential:
    """
    Builds a sequential mlp module.

    Args:
        layers: A list of integers representing the dimensions of each layer in the network.
        activation: The activation function to use between layers. Default is ReLU.
        batch_norm: Whether to use batch normalization between layers. Default is False.
        dropout: The probability of dropout between layers. Default is 0.0.

    Returns:
        A sequential neural network module with the specified architecture.

    Example usage:
    >>> net = build_network([10, 20, 30], nn.Tanh(), batch_norm=True, dropout=0.5)
    """
    layers_list = []
    for i in range(1, len(layers)):
        layers_list.append(nn.Linear(layers[i-1], layers[i]))
        if batch_norm:
            layers_list.append(nn.BatchNorm1d(layers[i]))
        layers_list.append(activation)
        if dropout > 0.0:
            layers_list.append(nn.Dropout(dropout))
    return nn.Sequential(*layers_list)

class Encoder(nn.Module):
    """
    Encoder module for Adversarial Autoencoder.

    Args:
        dims (list): List of dimensions for input, hidden layers and latent code in the form [input_dim, [h_dim1, h_dim2, ...], z_dim].
        activation (nn.Module): Activation function to use for hidden layers. Default is ReLU.
        batch_norm (bool): Whether to apply batch normalization to hidden layers. Default is False.
        dropout (float): Dropout probability for hidden layers. Default is 0.0.

    Example usage:
    >>> encoder = Encoder([784, [512, 256], 64], activation=nn.LeakyReLU(0.2), batch_norm=True, dropout=0.5)
    """

    def __init__(self, dims, hidden_activation: nn.Module = nn.ReLU(),
                  batch_norm: bool = False, dropout: float = 0.0):
        super(Encoder, self).__init__()

        [x_dim, h_dims, z_dim] = dims
        self.hidden_dims = [x_dim] + h_dims
        self.hidden_activation = hidden_activation
        self.hidden = build_network(self.hidden_dims, activation=hidden_activation, batch_norm=batch_norm, dropout=dropout)
        self.generate_z = nn.Linear(self.hidden_dims[-1], z_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hidden(x)
        return self.generate_z(x)

class Decoder(nn.Module):
    """
    Decoder module for Adversarial Autoencoder.

    Args:
        dims (list): List of dimensions for latent code, hidden layers and output.
        hidden_activation (nn.Module, optional): Activation function to use for hidden layers. Defaults to nn.ReLU().
        batch_norm (bool, optional): Whether to apply batch normalization to hidden layers. Defaults to False.
        dropout (float, optional): Dropout probability for hidden layers. Defaults to 0.0.
        output_activation (nn.Module, optional): Activation function to use for output layer. Defaults to None.

    Returns:
        nn.Module: Decoder module with the specified architecture.

    Example usage:
    >>> decoder = Decoder([64, [128, 128], 784], nn.Tanh(), batch_norm=True, dropout=0.5, output_activation=nn.Sigmoid())
    """

    def __init__(self, dims, hidden_activation: nn.Module = nn.ReLU(),
                 batch_norm: bool = False, dropout: float = 0.0, output_activation: nn.Module = None):
        super(Decoder, self).__init__()

        [z_dim, h_dims, x_dim] = dims
        self.hidden_dims = [z_dim] + h_dims
        self.hidden_activation = hidden_activation
        self.hidden = build_network(self.hidden_dims, activation=hidden_activation, batch_norm=batch_norm, dropout=dropout)
        self.recon = nn.Linear(self.hidden_dims[-1], x_dim)
        self.output_activation = output_activation

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.hidden(z)
        if self.output_activation is not None:
            return self.output_activation(self.recon(x))
        else:
            return self.recon(x)

class Discriminator(nn.Module):
    """
    Discriminator network architecture.

    Args:
    - hidden_dims (List): A list of integers representing the number of neurons in each hidden layer.
    - hidden_activation (nn.Module): The activation function to use in the hidden layers. Default is ReLU.
    - batch_norm (bool): Whether to use batch normalization after each hidden layer. Default is False.
    - dropout (float): The probability of dropout. Default is 0.0.
    - out_activation (str): The activation function to use in the output layer. Either 'sigmoid' or 'softmax'. Default is 'sigmoid'.
        
    Example usage:
    >>> D = Discriminator(dims=[10, [256, 128]], hidden_activation=nn.LeakyReLU(0.2), batch_norm=True, dropout=0.5, out_activation='sigmoid')
    """
    
    def __init__(self, dims, hidden_activation: nn.Module = nn.ReLU(), batch_norm: bool = False, 
                 dropout: float = 0.0, out_activation: str = 'Sigmoid'):
        super(Discriminator, self).__init__()
        [z_dim, D_dims] = dims
        self.hidden_dims = [z_dim] + D_dims
        self.hidden_activation = hidden_activation
        self.out_activation = out_activation

        self.hidden_layers = build_network(self.hidden_dims, activation=hidden_activation, batch_norm=batch_norm, dropout=dropout)
        
        if out_activation == 'Sigmoid':
            self.out_layer = nn.Sequential(nn.Linear(self.hidden_dims[-1], 1), nn.Sigmoid())
        elif out_activation == 'Softmax':
            self.out_layer = nn.Sequential(nn.Linear(self.hidden_dims[-1], 2), nn.Softmax(dim=1))
        else:
            raise ValueError("Invalid out_activation parameter. Must be 'Sigmoid' or 'Softmax'.")
        
    def forward(self, x):
        x = self.hidden_layers(x)
        x = self.out_layer(x)
        return x