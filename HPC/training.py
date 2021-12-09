#!/usr/bin/python3

import numpy as np
from torchsummary import summary
import pickle
import torch
import datetime
from torch.utils.tensorboard import SummaryWriter
from VAE import VAE

f=open('./FFT_AllSubject_Training_Face_minmaxNorm','rb')
fftdata=pickle.load(f)

result = np.zeros((1,3,102,188))
for l in range(len(fftdata)):
    fftdata[l]=np.where(fftdata[l]>1,1,fftdata[l])
    #totalsample+=fftdata[i].shape[0]
    result = np.vstack((result,fftdata[l]))

## Training
coder = VAE(3,3)
print(f'Device : {coder.device}')
bceloss=torch.nn.BCELoss()
#print(fftdata[1][111,0,0])
epoch=100
batchsize=10
batch=result.shape[0]//10
#tb = SummaryWriter('Training_CNS/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'_R2_100epoch')
for e in range(epoch):
    idxs=np.random.choice(result.shape[0],result.shape[0],replace=False)
    trainingloss=0
    rloss=0
    kloss=0
    for i in range(batch):
        idx=idxs[i*batchsize:i*batchsize+batchsize]
        data = torch.tensor(result[idx,:,:,:]).float().to(coder.device)
        out,mu,v = coder(data)
        regconstructionloss = bceloss(out,data)
        KLDloss = - 0.5 * torch.sum(1+ v - mu.pow(2) - v.exp())
            #print(f'R {regconstructionloss}')
            #print(f'K {KLDloss}')
        totalloss=regconstructionloss+KLDloss
        coder.optimizer.zero_grad()
        trainingloss+=totalloss.item()
        rloss=regconstructionloss.item()
        kloss=KLDloss.item()
        totalloss.backward()
        coder.optimizer.step()
            #print(totalloss.item())
    #tb.add_scalar("Training Avg Loss", (trainingloss/batch), e)
    #tb.add_scalar("Training Avg RLoss", (rloss/batch), e)
    #tb.add_scalar("Training Avg KLoss", (kloss/batch), e)
    #for name, weight in coder.named_parameters():
    #    tb.add_histogram('coder_'+name,weight, e)
    #    tb.add_histogram(f'coder_{name}.grad',weight.grad, e)
    print(f'epoch: {e:3} batch: {i:3} Training Batch Avg loss: {trainingloss/batch:10.8f}')
