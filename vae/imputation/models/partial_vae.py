
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal
from sklearn.model_selection import train_test_split
from copy import deepcopy
from torch.nn import BCELoss

class PartialVAE:
    def __init__(self, pointnet, encoder, decoder, optimizer, scheduler, config):
        self.pointnet = pointnet
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config

    @staticmethod
    def sample_from_normal(mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        return Normal(mu, torch.exp(0.5*log_sigma)).rsample((1,)).squeeze()

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

        best_pointnet = deepcopy(self.pointnet)
        best_encoder = deepcopy(self.encoder)
        best_decoder = deepcopy(self.decoder)
        best_loss = np.inf

        for epoch in range(self.config['max_epochs']):

            self.pointnet.train()
            self.encoder.train()
            self.decoder.train()

            for x_sample in tr_dataloader:
                m_sample = (~x_sample.isnan()).float().to(self.config['device'])                 # generate mask (0: missing 1: present)
                x_sample.nan_to_num_().to(self.config['device'])                                 # zero-out na values
                e_sample = self.pointnet(x_sample, m_sample)                                     # embedding through point net
                z_mu, z_log_sigma = self.encoder(e_sample)                                       # typical VAE-encoder

                z_sample = self.sample_from_normal(z_mu, z_log_sigma).to(self.config['device'])  # sampling with reparameterization trick
                x_mu, x_log_sigma = self.decoder(z_sample)                                       # typical VAE-decoder

                kld = -0.5 * (1 + z_log_sigma - z_mu.square() - z_log_sigma.exp())
                recon_nll = 0.5 * torch.square((x_sample - x_mu) / x_log_sigma.exp()) + x_log_sigma + 0.5 * np.log(2*np.pi)
                recon_nll *= m_sample

                kld = kld.sum()
                recon_nll = recon_nll.sum()
                loss = kld + recon_nll

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.scheduler.step()

            self.pointnet.eval()
            self.encoder.eval()
            self.decoder.eval()

            with torch.no_grad():
                epoch_loss = 0.
                for x_sample in val_dataloader:
                    m_sample = (~x_sample.isnan()).float().to(self.config['device'])
                    x_sample.nan_to_num_().to(self.config['device'])
                    e_sample = self.pointnet(x_sample, m_sample)
                    z_mu, z_log_sigma = self.encoder(e_sample)

                    z_sample = self.sample_from_normal(z_mu, z_log_sigma).to(self.config['device'])
                    x_mu, x_log_sigma = self.decoder(z_sample)

                    kld = -0.5 * (1 + z_log_sigma - z_mu.square() - z_log_sigma.exp())
                    recon_nll = 0.5 * torch.square((x_sample - x_mu) / x_log_sigma.exp()) + x_log_sigma + 0.5 * np.log(2 * np.pi)
                    recon_nll *= m_sample

                    kld = kld.sum()
                    recon_nll = recon_nll.sum()
                    loss = kld + recon_nll
                    epoch_loss += loss.item()

                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_pointnet = deepcopy(self.pointnet)
                    best_encoder = deepcopy(self.encoder)
                    best_decoder = deepcopy(self.decoder)

        self.pointnet = deepcopy(best_pointnet)
        self.encoder = deepcopy(best_encoder)
        self.decoder = deepcopy(best_decoder)


    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        dataloader = DataLoader(
            torch.from_numpy(x).to(self.config['device']).to(torch.float32),
            batch_size=self.config['batch_size'],
            shuffle=False, drop_last=False
        )

        self.pointnet.eval()
        self.encoder.eval()
        self.decoder.eval()
        x_recon = []
        with torch.no_grad():
            for x_sample in dataloader:
                m_sample = (~x_sample.isnan()).float().to(self.config['device'])
                x_sample.nan_to_num_().to(self.config['device'])
                e_sample = self.pointnet(x_sample, m_sample)
                z_mu, z_log_sigma = self.encoder(e_sample)

                z_sample = self.sample_from_normal(z_mu, z_log_sigma)

                x_mu, _ = self.decoder(z_sample)
                x_recon.append(x_mu.detach().cpu().numpy())

        return np.concatenate(x_recon, axis=0)
