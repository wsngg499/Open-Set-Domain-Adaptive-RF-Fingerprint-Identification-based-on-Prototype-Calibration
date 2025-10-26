import numpy as np
import torch
import torch.utils.data
from torch import nn
from torch.autograd import Variable

class FHVAE(nn.Module):
    def __init__(self, nmu2,options):
        super(FHVAE, self).__init__()
        self.input_dim = 512
        self.z2_dim=32
        self.mu2_lookup = nn.Embedding(nmu2, self.z2_dim)
        nc =4
        ndf = options['ndf']
        self.z1_pre_encoder = nn.Sequential(
            # input size. (nc) x 32 x 32
            nn.Conv1d(nc, ndf * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv1d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv1d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv1d(ndf * 8, ndf, 4, 1, 0, bias=False),
            nn.BatchNorm1d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        self.z2_pre_encoder = nn.Sequential(
            # input size. (nc) x 32 x 32
            nn.Conv1d(nc, ndf * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv1d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv1d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv1d(ndf * 8, ndf, 4, 1, 0, bias=False),
            nn.BatchNorm1d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )

        self.decoder = nn.Sequential(
            # input is Z, going into a convolution,(1-1)*s-2*p+k->0*1-2*0+4=4
            nn.ConvTranspose1d(ndf, ndf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(ndf * 8),
            nn.LeakyReLU(True),
            # state size. (ngf*8) x (4-1)*s-2p+4=12-2+4=16
            nn.ConvTranspose1d(ndf * 8, ndf * 4, 4, 4, 0, bias=False),
            nn.BatchNorm1d(ndf * 4),
            nn.LeakyReLU(True),
            # state size. (ngf*4) x (16-1)*4-2p+k=60-2*0+4=64
            nn.ConvTranspose1d(ndf * 4, ndf * 2, 4, 4, 0, bias=False),
            nn.BatchNorm1d(ndf * 2),
            nn.LeakyReLU(True),
            # state size. (ngf*2) x (64-1)*4-2p+k=252+4=256
            nn.ConvTranspose1d(ndf * 2, ndf, 4, 4, 0, bias=False),
            nn.BatchNorm1d(ndf),
            nn.LeakyReLU(True),
            # state size. (nc) x (256-1)*2-2p+k=512-2-2+4=512
            nn.ConvTranspose1d(ndf, nc, 4, 2, 1, bias=False),
            # nn.Sigmoid()
        )

        self.z1_linear = nn.Linear(ndf,ndf)
        self.z2_linear = nn.Linear(ndf,ndf)

    def encode(self, x, y):
        batch_size = x.size(0)
        T = x.size(1)
        mu2 = self.mu2_lookup(y)

        z2_pre_out= self.z2_pre_encoder(x)
        qz2_x = self.z2_linear(z2_pre_out.view(z2_pre_out.size(0), -1))
        z2_mu, z2_logvar = torch.chunk(qz2_x, 2, dim=-1)
        qz2_x = [z2_mu, z2_logvar]
        z2_sample = self.reparameterize(z2_mu, z2_logvar)

        z2 = z2_sample.unsqueeze(1).repeat(1, T, 1)
        x_z2 = torch.cat([x, z2], dim=-1)
        z1_pre_out= self.z1_pre_encoder(x_z2)
        qz1_x = self.z1_linear(z1_pre_out.view(z1_pre_out.size(0), -1))
        z1_mu, z1_logvar = torch.chunk(qz1_x, 2, dim=-1)
        qz1_x = [z1_mu, z1_logvar]
        z1_sample = self.reparameterize(z1_mu, z1_logvar)

        return mu2, qz2_x, z2_sample, qz1_x, z1_sample


    def decode(self, z1, z2, x):
        batch_size = x.size(0)
        z1_z2 = torch.cat([z1, z2], dim=-1).unsqueeze(1)
        out, x_mu, x_logvar, x_sample = [], [], [], []

        out_t = self.decoder(z1_z2.resize(z1_z2.size(0),64,1))

        return out_t

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, xin, xout, y):
        mu2, qz2_x, z2_sample, qz1_x, z1_sample = self.encode(xin, y)
        x_pre_out= self.decode(z1_sample, z2_sample, xout)
        return mu2, qz2_x, z2_sample, qz1_x, z1_sample,x_pre_out

def log_gauss(x, mu, logvar):
    log_2pi = torch.FloatTensor([np.log(2 * np.pi)]).cuda()
    return -0.5 * (log_2pi + logvar.data + torch.pow(x.data - mu.data, 2) / torch.exp(logvar.data))

def kld(p_mu, p_logvar, q_mu, q_logvar):
    return -0.5 * (1 + p_logvar.data - q_logvar.data - (torch.pow(p_mu.data - q_mu.data, 2) + torch.exp(p_logvar.data)) / torch.exp(q_logvar.data))


