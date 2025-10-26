## reference code is https://github.com/pytorch/examples/blob/master/dcgan/main.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class _netD32(nn.Module):
    def __init__(self, ngpu, nc, ndf):
        super(_netD32, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
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
            nn.Conv1d(ndf * 8, ndf * 16, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.avgpool = nn.AdaptiveAvgPool1d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(ndf * 16, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        output = self.avgpool(output)
        output = torch.flatten(output, 1)
        output = self.classifier(output).flatten()

        return output

class _netG32(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):
        super(_netG32, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose1d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm1d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose1d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose1d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose1d(ngf * 2, nc, 4, 2, 1, bias=False),
            nn.Sigmoid()
            # state size. (nc) x 32 x 32
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        
        return output

def Generator32(n_gpu, nz, ngf, nc):
    model = _netG32(n_gpu, nz, ngf, nc)
    model.apply(weights_init)
    return model

def Discriminator32(n_gpu, nc, ndf):
    model = _netD32(n_gpu, nc, ndf)
    model.apply(weights_init)
    return model


class _netD(nn.Module):
    def __init__(self, ngpu, nc, ndf):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
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
            nn.Sigmoid()
        )
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(ndf, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        output = self.avgpool(output)
        output = torch.flatten(output, 1)
        output = self.classifier(output).flatten()

        return output

class _netG(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc,options):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        ndf=nz
        if options['dataset'] == 'RF_2019':
            self.main = nn.Sequential(
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
                nn.Tanh()
            )
        if options['dataset'] == 'RF_2018':
            self.main = nn.Sequential(
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
                nn.ConvTranspose1d(ndf, ndf, 4, 2, 1, bias=False),
                nn.BatchNorm1d(ndf),
                nn.LeakyReLU(True),
                # state size. (nc) x (512-1)*2-2p+k=1024-2-2+4=1024
                nn.ConvTranspose1d(ndf, nc, 4, 2, 1, bias=False),
                nn.Tanh()
            )
        elif options['dataset'] == 'RF_2016':
            self.main = nn.Sequential(
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
                # state size. (nc) x (64-1)*2-2p+k=128-2-2+4=128
                nn.ConvTranspose1d(ndf * 2, nc, 4, 2, 1, bias=False),
                nn.Tanh()
            )
        else:
            self.main = nn.Sequential(
                # input is Z, going into a convolution,(1-1)*s-2*p+k->0*1-2*0+5=5
                nn.ConvTranspose1d(ndf, ndf * 8, kernel_size=5, stride=1, padding=0, bias=False),
                nn.BatchNorm1d(ndf * 8),
                nn.LeakyReLU(True),
                # state size. (ngf*8) x (5-1)*s-2p+4=25-5+5=25
                nn.ConvTranspose1d(ndf * 8, ndf * 4, 5, 5, 0, bias=False),
                nn.BatchNorm1d(ndf * 4),
                nn.LeakyReLU(True),
                # state size. (ngf*4) x (25-1)*5-2p+k=125-5-2*0+5=125
                nn.ConvTranspose1d(ndf * 4, ndf * 2, 5, 5, 0, bias=False),
                nn.BatchNorm1d(ndf * 2),
                nn.LeakyReLU(True),
                # state size. (ngf*2) x (125-1)*2-2p+k=250-2-2+4=250
                nn.ConvTranspose1d(ndf * 2, ndf, 4, 2, 1, bias=False),
                nn.BatchNorm1d(ndf),
                nn.LeakyReLU(True),
                # state size. (nc) x (250-1)*2-2p+k=500-2-2+4=500
                nn.ConvTranspose1d(ndf,ndf, 4, 2, 1, bias=False),
                nn.BatchNorm1d(ndf),
                nn.LeakyReLU(True),
                # state size. (nc) x (500-1)*3-2p+k=1500-3-2*0+3=1500
                nn.ConvTranspose1d(ndf, ndf//2, 3, 3, 0, bias=False),
                nn.BatchNorm1d(ndf//2),
                nn.LeakyReLU(True),
                # state size. (nc) x (1500-1)*2-2p+k=3000-2-2*0+2=3000
                nn.ConvTranspose1d(ndf//2, nc, 2, 2, 0, bias=False),
                nn.Tanh()
            )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        
        return output

def Generator(n_gpu, nz, ngf, nc,options):
    model = _netG(n_gpu, nz, ngf, nc, options)
    model.apply(weights_init)
    return model

def Discriminator(n_gpu, nc, ndf):
    model = _netD(n_gpu, nc, ndf)
    model.apply(weights_init)
    return model