import numpy as np
from torchsummary import summary
import pickle
import torch
import datetime


def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.normal_(m.weight,mean=0.0, std=0.00001)
        #m.bias.data.fill_(0)
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.normal_(m.weight,mean=0.0, std=0.00001)
        #m.bias.data.fill_(0)
    if isinstance(m, torch.nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight,mean=0.0, std=0.00001)
        #m.bias.data.fill_(0)

class VAE(torch.nn.Module):
    def __init__(self,ek_size,dk_size,channel,device=None):
        super(VAE, self).__init__()
        if(device is None):
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        # input dim 3*102*188 = 57528
        # H = [ (HIn + 2×padding[0]−dilation[0]×(kernel_size[0]−1)−1)/s ] + 1
        #   =    [( 102 + 0 - 1x(2) -1 )/2] + 1
        # W = [ (WIn +2×padding[0]−dilation[0]×(kernel_size[0]−1)−1)/s ] + 1
        #   = [ ( 188 + 0 - 1x(2) -1 ) /2 ] + 1
        # ConvTranspose
        # Out =(In−1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1
        #     = 4x2 - 0 + 2 + 0 + 1 = 11
        #     = 9x2 - 0 + 2 + 0 + 1 = 21

        #self.FC1 = torch.nn.Linear(512*5*4,8192).to(self.device)
        self.FC1 = torch.nn.Linear(512*5*4,5120).to(self.device)
        self.FCM = torch.nn.Linear(5120,256).to(self.device)
        self.FCV = torch.nn.Linear(5120,256).to(self.device)
        self.relu = torch.nn.ReLU().to(self.device)

        self.encoder =  torch.nn.Sequential(
            torch.nn.Conv2d(channel, 64, ek_size, stride=1,padding=ek_size//2),  #  C=64,H=102,W=188
            #torch.nn.Conv2d(64, 64, ek_size, stride=1,padding=ek_size//2),  #  C=64,H=102,W=188
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, stride=(2,3)),                       #  C=64,H=50,W=93
            torch.nn.Conv2d(64, 128, ek_size,stride=1,padding=ek_size//2),   #  C=128,H=50,W=93
            #torch.nn.Conv2d(128, 128, ek_size,stride=1,padding=ek_size//2),  #  C=128,H=50,W=93
            #torch.nn.Conv2d(128, 128, ek_size,stride=1,padding=ek_size//2),  #  C=128,H=50,W=93
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, stride=(2,3)),                                             #  C=128,H=24,W=46
            torch.nn.Conv2d(128, 256, ek_size,stride=1,padding=ek_size//2),  #  C=256,H=24,W=46
            #torch.nn.Conv2d(256, 256, ek_size,stride=1,padding=ek_size//2),   #  C=256,H=24,W=46
            #torch.nn.Conv2d(256, 256, ek_size,stride=1,padding=ek_size//2),   #  C=256,H=24,W=46
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, stride=(2,3)),                                           #  C=256,H=11,W=22
            torch.nn.Conv2d(256, 512, ek_size,stride=1,padding=ek_size//2),  #  C=512,H=11,W=22
            #torch.nn.Conv2d(512, 512, ek_size,stride=1,padding=ek_size//2),   #  C=512,H=11,W=22
            #torch.nn.Conv2d(512, 512, ek_size,stride=1,padding=ek_size//2),   #  C=512,H=11,W=22
            torch.nn.MaxPool2d(3, stride=(2,3)),
            #self.activation
                                                     #  C=512,H=5,W=10 = 25600 - mu C[0:256] , v C[256:512]
        ).to(self.device)
        self.decoder =  torch.nn.Sequential(
            #torch.nn.Conv2d(16, 512, dk_size, stride=1,padding=dk_size//2),  #  C=16,H=5,W=5
            #torch.nn.Conv2d(128, 512, dk_size, stride=1,padding=dk_size//2),   #  C=32,H=5,W=5
            torch.nn.ConvTranspose2d(16,512,2,stride=3),        #   11
            torch.nn.ReLU(),
            #torch.nn.Conv2d(256, 256, dk_size,stride=1,padding=dk_size//2 ),   #  C=64,H=11,W=15
            #torch.nn.Conv2d(256, 256, dk_size,stride=1,padding=dk_size//2 ),  # C=128,H=11,W=15
            torch.nn.ConvTranspose2d(512,128,4,stride=2),      # 24
            torch.nn.ReLU(),
            #torch.nn.Conv2d(128, 64, dk_size,stride=1,padding=dk_size//2),   #  C=64,H=24,W=32
            #torch.nn.Conv2d(64, 64, dk_size,stride=1,padding=dk_size//2),    #  C=32,H=24,W=32
            torch.nn.ConvTranspose2d(128,64,4,stride=2),       # 50
            torch.nn.ReLU(),
            #torch.nn.Conv2d(32, 16, dk_size,stride=1,padding=dk_size//2),    #  C=16,H=50,W=97
            #torch.nn.Conv2d(16, 16, dk_size,stride=1,padding=dk_size//2),    #  C=16,H=50,W=97
            torch.nn.ConvTranspose2d(64,3,4,stride=2)       #  50 (49*2=98+3=102)

        ).to(self.device)
        self.encoder.apply(weights_init)
        self.decoder.apply(weights_init)
        torch.nn.init.normal_(self.FC1.weight,mean=0.0, std=0.00001)
        torch.nn.init.normal_(self.FCM.weight,mean=0.0, std=0.00001)
        torch.nn.init.normal_(self.FCV.weight,mean=0.0, std=0.00001)
        #self.FC1.bias.data.fill_(0)
        #self.FCM.bias.data.fill_(0)
        #self.FCV.bias.data.fill_(0)

    def reparametrize(self,mu,log_var):
        #Reparametrization Trick to allow gradients to backpropagate from the
        #stochastic part of the model
        sigma = torch.exp(0.5*log_var)
        z = torch.randn_like(log_var,device=self.device)
        #z= z.type_as(mu)
        return mu + sigma*z
    def forward(self, x):
        feature = self.encoder(x)
        feature = feature.view(-1,512*5*4)
        sampling = self.FC1(feature)
        sampling = self.relu(sampling)
        #print(f'sampling {torch.max(sampling)}')
        mu = self.FCM(sampling)
        v = self.FCV(sampling)
        out = self.reparametrize(mu,v)
        out = out.view(-1,16,4,4)
        out = self.decoder(out)
        return out,mu,v
