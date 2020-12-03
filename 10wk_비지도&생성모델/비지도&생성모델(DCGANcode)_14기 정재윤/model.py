'''
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html 참고
'''

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

image_size = 64 # image 한 변의 길이
batch_size = 128 # batch 크기
nc = 1 # input channel 개수
nz = 100 # z latent vector 크기
ngf = 64 # generator feature map 크기
ndf = 64 # discriminator feature map 크기
num_epochs = 5 # epoch 수
lr = 0.0002 # Adam learning rate
beta1 = 0.5 # Adam beta1


def make_reproducible(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    
def get_loader():
    download_root = 'data'
    mnist_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(), 
        transforms.Normalize((0.5), (0.5)),
    ])
    dataset = MNIST(download_root, transform=mnist_transform, train=True, download=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=0)
    return dataloader

def weights_init(m):
    '''
    Conv 레이어와 BatchNorm 레이어의 weights를 논문에 나온대로 initialize
    '''
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
        
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        '''
        ConvTranspose2d->BatchNorm2d->ReLU를 반복
        channel 개수는 1/2로 줄여나감
        bias=False는 BatchNorm2d 때문에 어차피 bias가 의미 없기 때문
        마지막 Tanh()는 output 값이 -1~1 사이로 나오도록 하기 위함
        '''
        self.main = nn.Sequential(
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size: [ngf*8 x 4 x 4]
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size: [ngf*4 x 8 x 8]
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size: [ngf*2 x 16 x 16]
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size: [(ngf) x 32 x 32]
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size: [nc x 64 x 64]
        )


    def forward(self, input):
        return self.main(input)
    
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        '''
        Conv2d->BatchNorm2d->LeakyReLU를 반복
        channel 개수는 2배로 늘려나감
        bias=False는 BatchNorm2d 때문에 어차피 bias가 의미 없기 때문
        마지막 Sigmoid()는 binary classification을 위해
        '''
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
    


if __name__ == '__main__':
    
    make_reproducible() # 재현 가능성
    
    loader = get_loader() # dataloader 가져오기
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # cuda나 cpu
    
    netG = Generator().to(device) # generator 만들기
    netG.apply(weights_init) # weight 초기화
    
    netD = Discriminator().to(device) # discriminator 만들기
    netD.apply(weights_init) # weight 초기화
    
    criterion = nn.BCELoss() # loss 정의
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999)) # discriminator optimizer 정의
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999)) # generator optimizer 정의
    
    # 학습 시작
    for epoch in range(num_epochs):
        
        errDs = []
        errGs = []
        D_x = []
        D_G_z1 = []
        D_G_z2 = []
        
        for i, (data, _) in enumerate(loader):
            
            b_size = data.size(0) # batch size
            
            # A. discriminator 학습
            
            # 1. 전부 real인 데이터들로 학습 (target: real)
            netD.zero_grad()
            label = torch.full((b_size,), 1.0, dtype=torch.float, device=device) # label은 전부 1 (real)
            output = netD(data.to(device)).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            
            D_x.append(output.mean().item()) # discriminator가 real을 real로 판단할 확률

            # 2. generator가 생성한 데이터들로 학습 (target: fake)
            noise = torch.randn(b_size, nz, 1, 1, device=device) # latent vector v 초기화
            fake = netG(noise) # fake 생성
            label.fill_(0.0) # label은 전부 0 (fake)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            optimizerD.step()
            
            D_G_z1.append(output.mean().item()) # discriminator가 fake를 real로 판단할 확률
            errD = errD_real + errD_fake
            errDs.append(errD.item())

            # B. generator 학습
            netG.zero_grad()
            label.fill_(1.0) # discriminator가 real(1)로 판단하길 원하므로
            output = netD(fake).view(-1) 
            errG = criterion(output, label)
            errG.backward()
            optimizerG.step()
            
            D_G_z2.append(output.mean().item()) # discriminator가 한번 업데이트 된 후 fake를 real로 판단할 확률
            errGs.append(errG.item())
            
        # 학습 경과 출력 (각 epoch의 평균 loss)
        print(f'[Epoch{epoch}의 평균] Loss_D: {np.mean(errDs)} | Loss_G: {np.mean(errGs)} | D(x): {np.mean(D_x)} | D(G(z)): {np.mean(D_G_z1)} / {np.mean(D_G_z2)}')
        

