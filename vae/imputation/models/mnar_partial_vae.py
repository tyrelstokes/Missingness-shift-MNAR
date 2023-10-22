
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal
from sklearn.model_selection import train_test_split
from copy import deepcopy
from torch.nn import BCELoss

class MNARPartialVAE:
    def __init__(self, pointnet, masknet, encoder, decoder, optimizer, scheduler, config):
        self.pointnet = pointnet
        self.masknet = masknet
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config

    @staticmethod
    def sample_from_normal(mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        # return mu + Normal(0, torch.exp(0.5*log_sigma)).rsample((1,)).squeeze()
        return Normal(mu, torch.exp(0.5 * log_sigma)).rsample((1,)).squeeze()

    def fit(self, x: np.ndarray) -> None:

        tr_x, val_x = train_test_split(x, test_size=0.2, random_state=0)
        tr_dataloader = DataLoader(
            torch.from_numpy(tr_x).to(self.config['device']).to(torch.float32),
            batch_size=self.config['batch_size'],
            shuffle=True, drop_last=False
        )

        val_dataloader = DataLoader(
            torch.from_numpy(val_x).to(self.config['device']).to(torch.float32),
            batch_size=self.config['batch_size'],
            shuffle=True, drop_last=False
        )

        bce_loss = BCELoss(reduction='mean')

        best_masknet = deepcopy(self. masknet)
        best_pointnet = deepcopy(self.pointnet)
        best_encoder = deepcopy(self.encoder)
        best_decoder = deepcopy(self.decoder)
        best_loss = np.inf

        self.masknet.train()
        self.pointnet.train()
        self.encoder.train()
        self.decoder.train()
        for epoch in range(self.config['max_epochs']):
            for x_sample in tr_dataloader:
                m_sample = (~x_sample.isnan()).float()  # generate mask (0: missing 1: present)
                x_sample.nan_to_num_()  # zero-out na values
                e_sample = self.pointnet(x_sample, m_sample)  # embedding through point net (set encoder)
                z_mu, z_log_sigma = self.encoder(e_sample)  # typical VAE-encoder

                z_sample = self.sample_from_normal(z_mu, z_log_sigma)  # sampling with reparameterization trick
                x_mu, x_log_sigma = self.decoder(z_sample)  # typical VAE-decoder

                kld = -0.5 * (1 + z_log_sigma - z_mu.square() - z_log_sigma.exp())

                x_mixed = (1-m_sample)*x_mu + m_sample*x_sample
                dec_mean_mask = self.masknet(x_mixed)

                nll_mask = bce_loss(dec_mean_mask, m_sample)

                nll_recon = 0.5 * torch.square((x_sample - x_mu) / x_log_sigma.exp()) + x_log_sigma + 0.5 * np.log(
                    2 * np.pi)
                nll_recon *= m_sample
                nll = nll_recon + self.masknet.mask_variables * nll_mask

                kld = kld.sum()
                nll = nll.sum()
                loss = kld + nll

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.masknet.mask_variables.data = nn.Sigmoid()(self.masknet.mask_variables.data) # to ensure the mask variables to take values between 0 and 1

            self.scheduler.step()

            self.masknet.eval()
            self.pointnet.eval()
            self.encoder.eval()
            self.decoder.eval()

            with torch.no_grad():
                epoch_loss = 0.
                for x_sample in val_dataloader:
                    m_sample = (~x_sample.isnan()).float()  # generate mask (0: missing 1: present)
                    x_sample.nan_to_num_()  # zero-out na values
                    e_sample = self.pointnet(x_sample, m_sample)  # embedding through point net (set encoder)
                    z_mu, z_log_sigma = self.encoder(e_sample)  # typical VAE-encoder

                    z_sample = self.sample_from_normal(z_mu, z_log_sigma)  # sampling with reparameterization trick
                    x_mu, x_log_sigma = self.decoder(z_sample)  # typical VAE-decoder

                    kld = -0.5 * (1 + z_log_sigma - z_mu.square() - z_log_sigma.exp())

                    x_mixed = (1 - m_sample) * x_mu + m_sample * x_sample
                    dec_mean_mask = self.masknet(x_mixed)

                    nll_mask = bce_loss(dec_mean_mask, m_sample)

                    nll_recon = 0.5 * torch.square((x_sample - x_mu) / x_log_sigma.exp()) + x_log_sigma + 0.5 * np.log(2 * np.pi)
                    nll_recon *= m_sample
                    nll = nll_recon + self.masknet.mask_variables * nll_mask

                    kld = kld.sum()
                    nll = nll.sum()
                    loss = kld + nll
                    epoch_loss += loss.item()

                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_masknet = deepcopy(self.masknet)
                    best_pointnet = deepcopy(self.pointnet)
                    best_encoder = deepcopy(self.encoder)
                    best_decoder = deepcopy(self.decoder)

        self.masknet = deepcopy(best_masknet)
        self.pointnet = deepcopy(best_pointnet)
        self.encoder = deepcopy(best_encoder)
        self.decoder = deepcopy(best_decoder)


    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        dataloader = DataLoader(
            torch.from_numpy(x).to(self.config['device']).to(torch.float32),
            batch_size=self.config['batch_size'],
            shuffle=False, drop_last=False
        )

        self.masknet.eval()
        self.pointnet.eval()
        self.encoder.eval()
        self.decoder.eval()
        x_recon = []
        with torch.no_grad():
            for x_sample in dataloader:
                m_sample = (~x_sample.isnan()).float()
                x_sample.nan_to_num_()
                e_sample = self.pointnet(x_sample, m_sample)
                z_mu, z_log_sigma = self.encoder(e_sample)

                z_sample = self.sample_from_normal(z_mu, z_log_sigma)

                x_mu, _ = self.decoder(z_sample)
                x_recon.append(x_mu.detach().cpu().numpy())

        return np.concatenate(x_recon, axis=0)
