#!/usr/bin/env python
"""
Author: Zhiyuan Chu
Date: March 27, 2023
File Name: model.py
"""

import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import pytorch_lightning as pl

from .layer import build_network, Encoder, Decoder, Discriminator

class LitAAE(pl.LightningModule):
    """
    A PyTorch Lightning Module implementing the Adversarial Autoencoder (AAE).
    
    Args:
        dims : A list of dimensions for the input, latent, encoder, decoder, and discriminator layers.
        encoder_act (nn.Module): Activation function for the encoder.
        decoder_act (nn.Module): Activation function for the decoder.
        discriminator_act (nn.Module): Activation function for the discriminator.
        batch_norm (bool): Whether to use batch normalization.
        dropout (float): Dropout rate.
        decoder_out (nn.Module): Activation function for the output layer of the decoder.
        D_out (str): Output activation function for the discriminator. Either 'Sigmoid' or 'Softmax'.
        lr (float): Learning rate for the optimizer.
        b1 (float): Beta1 parameter for Adam optimizer.
        b2 (float): Beta2 parameter for Adam optimizer.
        batch_size (int): Batch size.
    """
    
    
    def __init__(self, dims,
                 encoder_act = nn.ReLU(),
                 decoder_act = nn.ReLU(),
                 discriminator_act = nn.ReLU(), 
                 batch_norm: bool = False, 
                 dropout: float = 0., 
                 decoder_out = None, 
                 D_out = 'Sigmoid',
                 lr: float = 0.0002,
                 b1: float = 0.9,
                 b2: float = 0.999,
                 batch_size: int = 64
                ):
        
        super(LitAAE, self).__init__()
        self.save_hyperparameters(ignore=['encoder_act', 'decoder_act', 'discriminator_act'])
        self.automatic_optimization = False
        
        [x_dim, z_dim, encoder_dims, decoder_dims, D_dims] = dims
        # D_dims: discriminator dims
        # e.g. [3000, 10, [64, 64], [32, 16], [32]]
        
        self.latent_dim = z_dim
        self.D_act = D_out
        self.encoder = Encoder([x_dim, encoder_dims, z_dim], 
                               hidden_activation = encoder_act, 
                               batch_norm = batch_norm, 
                               dropout = dropout
                              )
        self.decoder = Decoder([z_dim, decoder_dims, x_dim], 
                               hidden_activation = decoder_act, 
                               batch_norm = batch_norm, 
                               dropout = dropout, 
                               output_activation = decoder_out
                              )
        self.discriminator = Discriminator([z_dim, D_dims], 
                                           hidden_activation = discriminator_act, 
                                           batch_norm = batch_norm, 
                                           dropout = dropout, 
                                           out_activation = D_out
                                          )
        self.reset_parameters()
        
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                #init.kaiming_normal_(m.weight.data, nonlinearity = 'relu')
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def forward(self, x):
        z = self.encoder(x)
        recon_x = self.decoder(z)
        return recon_x
    
    def predict(self, x):
        z = self.encoder(x)
        pred = self.discriminator(z)
        return pred
    
    def get_latent(self, x):
        z = self.encoder(x)
        return z
    
    def adversarial_loss(self, x_hat, x):
        loss = None
        if self.D_act == 'Sigmoid':
            loss = F.binary_cross_entropy(x_hat, x)
        if self.D_act == 'Softmax':
            loss = F.cross_entropy(x_hat, x)
        if loss is None:
            raise ValueError("Invalid D_act value: {}".format(self.D_act))
        return loss
    
    def reconstruction_loss(self, x):
        z = self.encoder(x)
        recon_x = self.decoder(z)
        loss = F.mse_loss(recon_x, x)
        self.log("recon_loss", loss, prog_bar=True, sync_dist=True)
        return loss
    
    def get_D_loss(self, x):
        z_gaussian = torch.randn(x.size(0), self.latent_dim).type_as(x)
        valid = torch.ones(x.size(0), 1).type_as(x)
        real_loss = self.adversarial_loss(self.discriminator(z_gaussian), valid)

        z_generated = self.encoder(x)
        fake = torch.zeros(x.size(0), 1).type_as(x)
        fake_loss = self.adversarial_loss(self.discriminator(z_generated.detach()), fake)
            
        d_loss = (real_loss + fake_loss) / 2
        self.log("d_loss", d_loss, prog_bar=True, sync_dist=True)
        return d_loss
    
    def get_G_loss(self, x):
        z_generated = self.encoder(x)
        validity_generated = self.discriminator(z_generated)
        valid = torch.ones(x.size(0), 1)
        valid = valid.type_as(x)
        g_loss = self.adversarial_loss(validity_generated, valid)
        self.log("g_loss", g_loss, prog_bar=True, sync_dist=True)
        return g_loss

    def training_step(self, batch, batch_idx):
        x, _ = batch
        gen_opt, disc_opt = self.optimizers()
        
        gen_opt.zero_grad()
        recon_loss = self.reconstruction_loss(x)
        adver_loss = self.get_G_loss(x)
        g_loss = 0.001 * adver_loss + 0.999 * recon_loss
        g_loss.backward()
        gen_opt.step()
        
        disc_opt.zero_grad()
        d_loss = self.get_D_loss(x)
        d_loss.backward()
        disc_opt.step()

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        gen_opt = torch.optim.Adam(itertools.chain(self.encoder.parameters(), self.decoder.parameters()), lr=lr, betas=(b1, b2))
        disc_opt = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return gen_opt, disc_opt