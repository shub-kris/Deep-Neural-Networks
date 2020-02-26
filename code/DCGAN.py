import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

data_path = ".\datasets\celeba"
batch_size = 128
img_size = 64
nc = 3
nz = 100
ngf = 64
ndf = 64
lr = 0.0002
beta1 = 0.5
num_epochs = 5

dataset = dset.ImageFolder(root=data_path,
                           transform=transforms.Compose([
                               transforms.Resize(img_size),
                               transforms.CenterCrop(img_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
                               
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True)


# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.convG = nn.Sequential(
            nn.ConvTranspose2d(in_channels = nz, out_channels = ngf * 8, kernel_size = 4, bias = False),
            nn.BatchNorm2d(ngf* 8),
            nn.ReLU(inplace = True),
            
            nn.ConvTranspose2d(ngf * 8, ngf* 4, 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(ngf* 4),
            nn.ReLU(inplace = True),
            
            nn.ConvTranspose2d(ngf* 4, ngf*2, 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(inplace = True),
            
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace = True),
            
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias = False),
            nn.Tanh()
        )
            
        
    def forward(self, x):
        x = self.convG(x)
        return x
        
netG = Generator().to(device)
netG.apply(weights_init)
print(netG)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.convD = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias = False),
            nn.LeakyReLU(0.2, inplace = True),
            
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace = True),
            
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace = True),
            
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ndf* 8),
            nn.LeakyReLU(inplace = True),
            
            nn.Conv2d(ndf* 8, 1, 4, 2, 0, bias = False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.convD(x)
        return x
    
netD = Discriminator().to(device)
netD.apply(weights_init)
print(netD)

criterion = nn.BCELoss()

noise = torch.randn(batch_size, nz, 1, 1, device = device)   #(64,100,1,1)

real_label = 1
fake_label = 0

optimizerD = optim.Adam(netD.parameters(), lr = lr, betas = (beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr = lr, betas = (beta1, 0.999))

G_losses = []
D_losses = []

for epoch in range(num_epochs):
    
    for i, data in enumerate(dataloader):
        
        netD.zero_grad()
        
        real_x = data[0].to(device)
        b_size = real_x.size(0)
        label = torch.full((b_size,), real_label, device = device)
        
        output = netD(real_x).view(-1)
        
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()
        
        
        noise = torch.randn(b_size, nz, 1, 1, device = device)
        fake_x = netG(noise)
        label.fill_(fake_label)
        
        output = netD(fake_x.detach()).view(-1)
        
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        
        errD = errD_fake + errD_real
        
        optimizerD.step()
        
        netG.zero_grad()
        label.fill_(real_label)
        output = netD(fake_x).view(-1)
        
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        
        optimizerG.step()
        
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, num_epochs, i, len(dataloader),
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        
        if i % 100 == 0:
            vutils.save_image(real_x,
                    '%s/real_samples.png' % '.',
                    normalize=True)
                    
            fake = netG(noise)
            
            vutils.save_image(fake.detach(),
                    '%s/fake_samples_epoch_%03d.png' % ('.', epoch),
                    normalize=True)

