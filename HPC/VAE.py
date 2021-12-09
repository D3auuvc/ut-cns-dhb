import numpy as np
from torchsummary import summary
import pickle
import torch
import datetime
from torch.utils.tensorboard import SummaryWriter

class VAE(torch.nn.Module):
    def __init__(self,k_size,channel):
        super(VAE, self).__init__()
        self.device=None
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        # input dim 3*102*188 = 57528
        # H = [ (HIn + 2×padding[0]−dilation[0]×(kernel_size[0]−1)−1)/s ] + 1
        #   =    [( 102 + 0 - 1x(2) -1 )/2] + 1 
        # W = [ (WIn +2×padding[0]−dilation[0]×(kernel_size[0]−1)−1)/s ] + 1
        #   = [ ( 188 + 0 - 1x(2) -1 ) /2 ] + 1
        # ConvTranspose 
        # Out =(In−1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1
        #     = 4x2 - 0 + 2 + 0 + 1 = 11
        #     = 9x2 - 0 + 2 + 0 + 1 = 21
        self.activation = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(3, stride=2)
        
        self.encoder =  torch.nn.Sequential(
            torch.nn.Conv2d(channel, 64, k_size, stride=1,padding=1),  #  C=64,H=102,W=188
            torch.nn.Conv2d(64, 64, k_size, stride=1,padding=1),  #  C=64,H=102,W=188
            self.activation,
            self.maxpool,                                         #  C=64,H=50,W=93
            torch.nn.Conv2d(64, 128, k_size,stride=1,padding=1),   #  C=128,H=50,W=93
            torch.nn.Conv2d(128, 128, k_size,stride=1,padding=1),  #  C=128,H=50,W=93
            torch.nn.Conv2d(128, 128, k_size,stride=1,padding=1),  #  C=128,H=50,W=93
            self.activation,
            self.maxpool,                                          #  C=128,H=24,W=46
            torch.nn.Conv2d(128, 256, k_size,stride=1,padding=1),  #  C=256,H=24,W=46
            torch.nn.Conv2d(256, 256, k_size,stride=1,padding=1),   #  C=256,H=24,W=46
            torch.nn.Conv2d(256, 256, k_size,stride=1,padding=1),   #  C=256,H=24,W=46
            self.activation,
            self.maxpool,                                          #  C=256,H=11,W=22
            torch.nn.Conv2d(256, 512, k_size,stride=1,padding=1),  #  C=512,H=11,W=22
            torch.nn.Conv2d(512, 512, k_size,stride=1,padding=1),   #  C=512,H=11,W=22
            torch.nn.Conv2d(512, 512, k_size,stride=1,padding=1),   #  C=512,H=11,W=22
            self.activation,
            self.maxpool                                          #  C=512,H=5,W=10 = 25600 - mu C[0:256] , v C[256:512]
        ).to(self.device)
        self.decoder =  torch.nn.Sequential(
            torch.nn.Conv2d(256, 16, k_size, stride=1,padding=1),  #  C=16,H=5,W=10
            torch.nn.Conv2d(16, 32, k_size, stride=1,padding=1),   #  C=32,H=5,W=10
            torch.nn.ConvTranspose2d(32,32,(3,4),stride=2),        #  C=32,H=11,W=22
            self.activation,
            torch.nn.Conv2d(32, 64, k_size,stride=1,padding=1 ),   #  C=64,H=11,W=22
            torch.nn.Conv2d(64, 128, k_size,stride=1,padding=1 ),  # C=128,H=11,W=22
            torch.nn.ConvTranspose2d(128,128,(4,4),stride=2),      # C=128,H=24,W=46
            self.activation,
            torch.nn.Conv2d(128, 256, k_size,stride=1,padding=1),   #  C=64,H=24,W=46
            torch.nn.Conv2d(256, 32, k_size,stride=1,padding=1),    #  C=32,H=24,W=46
            torch.nn.ConvTranspose2d(32,32,(4,3),stride=2),        #  C=32,H=50,W=93
            self.activation,
            torch.nn.Conv2d(32, 16, k_size,stride=1,padding=1),    #  C=16,H=50,W=93
            torch.nn.Conv2d(16, 16, k_size,stride=1,padding=1),    #  C=16,H=50,W=93
            torch.nn.ConvTranspose2d(16,3,(4,4),stride=2),          #  C=3,H=102,W=188
            torch.nn.Sigmoid()
        ).to(self.device)
        
        self.parameters = set()
        self.parameters |= set(self.encoder.parameters())
        self.parameters |= set(self.decoder.parameters())
        self.optimizer = torch.optim.Adam(self.parameters, lr=0.001)
        
    def reparametrize(self,mu,log_var):
        #Reparametrization Trick to allow gradients to backpropagate from the 
        #stochastic part of the model
        sigma = torch.exp(0.5*log_var)
        z = torch.randn(mu.size(0),mu.size(1),mu.size(2),mu.size(3),device=self.device)
        z= z.type_as(mu)
        return mu + sigma*z
        
    def forward(self, x):
        out = self.encoder(x)
        mu = out[:,:256,:,:]
        v = out[:,256:,:,:]
        #print(mu)
        #print(v)
        out = self.reparametrize(mu,v)
        out = self.decoder(out)
        #print(mu)
        return out,mu,v
