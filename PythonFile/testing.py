#!/usr/bin/python3
import numpy as np
from torchsummary import summary
import pickle
import torch
import datetime
from torch.utils.tensorboard import SummaryWriter
from VAE_RawVsFFT import VAE
## Training

def loadData(path,W):
    f=open(path,'rb')
    data=pickle.load(f)
    result = np.zeros((1,3,102,W))
    for i in range(len(data)):
        data[i]=np.where(data[i]>1,1,data[i])
        data[i]=np.where(data[i]<0,0,data[i])
        result = np.vstack((result,data[i]))
    return result[1:]

result = loadData('./RAW_AllSubject_Training_Face_minmaxNorm',375)
fft_result = loadData('./FFT_AllSubject_Training_Face_minmaxNorm',188)
coder = torch.load('./Model_CNS/R47/bmodel_0.17104576771140945_epoch_98')
loss = torch.nn.BCEWithLogitsLoss()
tb = SummaryWriter('Testing_CNS/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'_R9_RAWvsFFT_TrainingR47_OnlyFaceTesting')
idx=[x for x in range(result.shape[0])]
batchsize=20
batch=result.shape[0]//batchsize
vloss=0
data = np.zeros((20,3,102,375))
for i in range(20):
    data[i]=np.random.rand(3,102,375)

for i in range(batch):
    bvloss=0
    coder.eval()
    bidx=idx[i*batchsize:i*batchsize+batchsize]
    #data = torch.tensor(result[bidx,:,:,0:102]).float().to(coder.device)
    data = torch.tensor(data).float().to(coder.device)
    #fdata = torch.tensor(fft_result[bidx,:,:,0:102]).float().to(coder.device)
    out,mu,v = coder(data)
    #regconstructionloss = loss(out,fdata)
    #vloss+=regconstructionloss.item()
    #bvloss+=regconstructionloss.item()
    out=out.detach()
    for k in range(out.shape[0]):
        tb.add_image("10 test examples batch_"+str(i) , torch.logit(out[k,:,:,:]) , k  , dataformats='CHW')
    mu=mu.detach()
    v=v.detach()
    mu = torch.median(mu)
    mq25 = torch.quantile(mu,0.25)
    mq75 = torch.quantile(mu,0.75)
    mv = torch.median(mu)
    vq25 = torch.quantile(v,0.25)
    vq75 = torch.quantile(v,0.75)
    tb.add_scalars("Testing mu Statistic", {'median' : mu,'q25' : mq25,'q75' : mq75}, i)
    tb.add_scalars("Testing v Statistic", {'median' : mv,'q25' : vq25,'q75' : vq75}, i)
    tb.add_scalar("Testing Avg batch Loss", (bvloss/batchsize), i)

#print(f'Testing Avg loss: {vloss/batch:10.8f}')
