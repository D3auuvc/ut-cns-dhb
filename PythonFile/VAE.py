import numpy as np
from torchsummary import summary
import pickle
import torch
import datetime


def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        #m.bias.data.fill_(0.01)
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        #m.bias.data.fill_(0.01)
        

class VAE(torch.nn.Module):
    def __init__(self,ek_size,dk_size,channel):
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
        self.pooling = torch.nn.AvgPool2d(3, stride=2)
        
        self.encoder =  torch.nn.Sequential(
            torch.nn.Conv2d(channel, 64, ek_size, stride=1,padding=ek_size//2),  #  C=64,H=102,W=188
            #torch.nn.Conv2d(64, 64, k_size, stride=1,padding=1),  #  C=64,H=102,W=188
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, stride=2),                       #  C=64,H=50,W=93
            torch.nn.Conv2d(64, 128, ek_size,stride=1,padding=ek_size//2),   #  C=128,H=50,W=93
            torch.nn.Conv2d(128, 128, ek_size,stride=1,padding=ek_size//2),  #  C=128,H=50,W=93
            #torch.nn.Conv2d(128, 128, k_size,stride=1,padding=1),  #  C=128,H=50,W=93
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, stride=2),                                             #  C=128,H=24,W=46
            torch.nn.Conv2d(128, 256, ek_size,stride=1,padding=ek_size//2),  #  C=256,H=24,W=46
            torch.nn.Conv2d(256, 256, ek_size,stride=1,padding=ek_size//2),   #  C=256,H=24,W=46
            #torch.nn.Conv2d(256, 256, k_size,stride=1,padding=1),   #  C=256,H=24,W=46
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, stride=2),                                           #  C=256,H=11,W=22
            torch.nn.Conv2d(256, 512, ek_size,stride=1,padding=ek_size//2),  #  C=512,H=11,W=22
            torch.nn.Conv2d(512, 512, ek_size,stride=1,padding=ek_size//2),   #  C=512,H=11,W=22
            #torch.nn.Conv2d(512, 512, k_size,stride=1,padding=1),   #  C=512,H=11,W=22
            torch.nn.MaxPool2d(3, stride=2)
            #self.activation
                                                     #  C=512,H=5,W=10 = 25600 - mu C[0:256] , v C[256:512]
        ).to(self.device)
        self.decoder =  torch.nn.Sequential(
            torch.nn.Conv2d(512, 16, dk_size, stride=1,padding=dk_size//2),  #  C=16,H=5,W=5
            #torch.nn.Conv2d(16, 32, k_size, stride=1,padding=1),   #  C=32,H=5,W=5
            torch.nn.ConvTranspose2d(16,32,3,stride=2),        #  C=32,H=11,W=15
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, dk_size,stride=1,padding=dk_size//2 ),   #  C=64,H=11,W=15
            #torch.nn.Conv2d(64, 128, k_size,stride=1,padding=1 ),  # C=128,H=11,W=15
            torch.nn.ConvTranspose2d(64,128,4,stride=2),      # C=128,H=24,W=32
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, dk_size,stride=1,padding=dk_size//2),   #  C=64,H=24,W=32
            #torch.nn.Conv2d(256, 64, k_size,stride=1,padding=1),    #  C=32,H=24,W=32
            torch.nn.ConvTranspose2d(256,64,4,stride=2),           #  C=32,H=50,W=97
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 32, dk_size,stride=1,padding=dk_size//2),    #  C=16,H=50,W=97
            #torch.nn.Conv2d(16, 16, k_size,stride=1,padding=1),    #  C=16,H=50,W=97
            torch.nn.ConvTranspose2d(32,3,4,stride=2)       #  C=3,H=102,W=188
            
            #torch.nn.Sigmoid()
        ).to(self.device)
        self.encoder.apply(weights_init)
        self.decoder.apply(weights_init)
        
        
    def reparametrize(self,mu,log_var):
        #Reparametrization Trick to allow gradients to backpropagate from the 
        #stochastic part of the model
        sigma = torch.exp(0.5*log_var)
        z = torch.randn_like(log_var,device=self.device)
        #z= z.type_as(mu)
        return mu + sigma*z
        
    def forward(self, x):
        feature = self.encoder(x)
        #mu = out[:,:,:,0:5]
        #v =  out[:,:,:,5:10]
        #out = self.reparametrize(mu,v)
        out = self.decoder(feature)
        return out,feature#,mu,v
        
