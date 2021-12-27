#!/usr/bin/python3
import numpy as np
from torchsummary import summary
import pickle
import torch
import datetime
from torch.utils.tensorboard import SummaryWriter
from VAE_RawVsFFT_flattenFeature import VAE
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

result = loadData('./FFT_AllSubject_Training_AllClass_minmaxNorm',188)
raw_result = loadData('./RAW_AllSubject_Training_AllClass_minmaxNorm',375)
coder = VAE(3,3,3)
coder = torch.load('./Model_CNS/R45/bmodel_0.1717244103550911_epoch_89')
summary(coder,(3,102,375))
loss=torch.nn.BCEWithLogitsLoss()
#print(fftdata[1][111,0,0])
epoch=500
batchsize=20
optimizer = torch.optim.Adam(coder.parameters(), lr=0.001)
tb = SummaryWriter('Training_CNS/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'_R48_bsize20_ksize3_Maxpool_4e4d_RawvsFFT_AllClass_flattenFeature256')
length=[x for x in range(result.shape[0])]
t=int(np.floor(result.shape[0]*0.7))
np.random.shuffle(length)
print(t)
tidxs=length[:t]
vidxs=length[t:]
#print(tidxs)
batch=t//batchsize
vbatch = (result.shape[0]-t)//batchsize
bestvloss=float('inf')
for e in range(epoch):
    trainingloss=0
    rloss=0
    kloss=0
    vloss=0
    vrloss=0
    vkloss=0
    coder.train()
    for i in range(batch):
        idx=tidxs[i*batchsize:i*batchsize+batchsize]
        #oidx=idx.copy()
        #np.random.shuffle(oidx)
        #print(oidx)
        #data = torch.tensor(raw_result[idx,:,:,:]).float().to(coder.device)
        data = torch.tensor(raw_result[idx,:,:,:]).float().to(coder.device)
        fdata = torch.tensor(result[idx,:,:,0:102]).float().to(coder.device)
        out,mu,v = coder(data)
        #print(torch.max(v))
        regconstructionloss = loss(out,fdata)
        KDloss = -0.5 * torch.mean( torch.sum(1+ v - mu.pow(2) - v.exp()))
        totalloss = regconstructionloss  + KDloss
        optimizer.zero_grad()
        trainingloss+=totalloss.item()
        rloss+=regconstructionloss.item()
        kloss += KDloss.item()
        totalloss.backward()
        optimizer.step()

    for name, weight in coder.encoder.named_parameters():
        tb.add_histogram('encoder'+name,weight, e)
        tb.add_histogram(f'encoder_{name}.grad',weight.grad, e)
           #print(weight.grad)
    for name, weight in coder.decoder.named_parameters():
        tb.add_histogram('decoder'+name,weight, e)
        tb.add_histogram(f'decoder_{name}.grad',weight.grad, e)
    coder.eval()
    mu=None
    v=None
    for i in range(vbatch):
        idx=vidxs[i*batchsize:i*batchsize+batchsize]
        #oidx=idx.copy()
        #np.random.shuffle(oidx)
        #print(idx)
        #data = torch.tensor(raw_result[idx,:,:,:]).float().to(coder.device)
        data = torch.tensor(raw_result[idx,:,:,:]).float().to(coder.device)
        fdata = torch.tensor(result[idx,:,:,0:102]).float().to(coder.device)
        out,mu,v = coder(data)
        regconstructionloss = loss(out,fdata)
        KDloss = -0.5 * torch.mean( torch.sum(1+ v - mu.pow(2) - v.exp()))
        totalloss = regconstructionloss  + KDloss
        vloss+=totalloss.item()
        vrloss+=regconstructionloss.item()
        vkloss+=KDloss.item()
        #print(totalloss.item())
    if(e%10==0):
        tb.add_image("10 val examples",torch.logit(out),e,dataformats='NCHW')
    if(vloss<bestvloss):
        torch.save(coder,'./Model_CNS/bmodel_'+str(vloss/vbatch)+'_epoch_'+str(e))
        bestvloss=vloss
    mu = mu.detach()
    v = v.detach()
    mmu = torch.median(mu)
    mq25 = torch.quantile(mu,0.25)
    mq75 = torch.quantile(mu,0.75)
    mv = torch.median(v)
    vq25 = torch.quantile(v,0.25)
    vq75 = torch.quantile(v,0.75)
    tb.add_scalars("Validation feature Statistic", {'median':mmu,'q25':mq25,'q75':mq75}, e)
    tb.add_scalars("Validation v Statistic", {'median':mv,'q25':vq25,'q75':vq75}, e)
    tb.add_scalar("Training Avg Loss", (trainingloss/batch), e)
    tb.add_scalar("Training Avg RLoss", (rloss/batch), e)
    tb.add_scalar("Training Avg KLoss", (kloss/batch), e)
    tb.add_scalar("Validation Avg Loss", (vloss/vbatch), e)
    tb.add_scalar("Validation Avg RLoss", (vrloss/vbatch), e)
    tb.add_scalar("Validation Avg KLoss", (vkloss/vbatch), e)

    print(f'epoch: {e:3} batch: {i:3} Training Batch Avg loss: {trainingloss/batch:10.8f} Validation Batch Avg loss: {vloss/vbatch:10.8f}  Validation Avg K loss: {vkloss/vbatch:10.8f},  Validation Avg R loss: {vrloss/vbatch:10.8f}')
