#!/usr/bin/python3
import numpy as np
from torchsummary import summary
import pickle
import torch
import datetime
from torch.utils.tensorboard import SummaryWriter
from VAE_Model.VAE_FFTvsFFT_flattenFeature102_Ksize4 import VAE
## Training

def loadData(path,W):
    f=open(path,'rb')
    data=pickle.load(f)
    if(W>1):
        result = np.zeros((1,3,102,W))
        for i in range(len(data)):
            data[i]=np.where(data[i]>1,1,data[i])
            data[i]=np.where(data[i]<0,0,data[i])
            result = np.vstack((result,data[i]))
        return result[1:]
    else:
        result = np.zeros((1,1))
        for i in range(len(data)):
            result = np.vstack((result,data[i]))
        return result[1:].reshape(-1,)

#fresult = loadData('./FFT_AllSubject_Training_Face_minmaxNorm',188)
fresult = loadData('./RawData/FFT_AllSubject_Training_AllClass_minmaxNorm',188)
Y = loadData('./RawData/FFT_AllSubject_Training_AllClass_Target',1)
#sresult = loadData('./FFT_AllSubject_Training_scramble_minmaxNorm',188)
#sraw_result = loadData('./RAW_AllSubject_Training_scramble_minmaxNorm',375)
coder = VAE(4,3,3)
#coder = torch.load('./Model_CNS/R95/bmodel_0.10959810337850026_epoch_437')
summary(coder,(3,102,188))
loss=torch.nn.BCEWithLogitsLoss()
#print(fftdata[1][111,0,0])
epoch=500
batchsize=50
optimizer = torch.optim.Adam(coder.parameters(), lr=0.001)
tb = SummaryWriter('Training_CNS_TensorBoard/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'_R108_bsize50_ksize4_Avgpool_5e6d_FFTVSFFT_noTransform_AllClass_flattenStackFeatureColumn102_reduceMnet_noinitWeight_withKLDloss')
length=[x for x in range(fresult.shape[0])]
t=int(np.floor(fresult.shape[0]*0.7))
np.random.shuffle(length)
print(t)
tidxs=length[:t]
vidxs=length[t:]
batch=t//batchsize
vbatch = (fresult.shape[0]-t)//batchsize
bestvloss=float('inf')
for e in range(epoch):
    trainingloss=0
    vloss=0
    rloss=0
    kloss=0
    vrloss=0
    vkloss=0
    coder.train()
    for i in range(batch):
        idx=tidxs[i*batchsize:i*batchsize+batchsize]
        #oidx=idx.copy()
        #np.random.shuffle(oidx)
        #print(oidx)
        #data = torch.tensor(raw_result[idx,:,:,:]).float().to(coder.device)
        data = torch.tensor(fresult[idx,:,:,:]).float().to(coder.device)
        out,mu,v = coder(data)
        #print(torch.max(v))
        regconstructionloss = loss(out,data)
        KDloss = -0.5 * torch.sum(1+ v - mu.pow(2) - v.exp())
        kloss += KDloss.item()
        #sKDloss = -0.5 * torch.mean( torch.sum(1+ sv - smu.pow(2) - sv.exp()))
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
        y = Y[idx]
        #oidx=idx.copy()
        #np.random.shuffle(oidx)
        #print(idx)
        #data = torch.tensor(raw_result[idx,:,:,:]).float().to(coder.device)
        data = torch.tensor(fresult[idx,:,:,:]).float().to(coder.device)
        out,mu,v = coder(data)
        regconstructionloss = loss(out,data)
        KDloss = -0.5 * torch.sum(1+ v - mu.pow(2) - v.exp())
        vkloss += KDloss.item()
        #sKDloss = -0.5 * torch.mean( torch.sum(1+ sv - smu.pow(2) - sv.exp()))
        totalloss = regconstructionloss + KDloss
        vloss+=totalloss.item()
        vrloss += regconstructionloss.item()
        out.detach().cpu()
        mu.detach().cpu()
        v.detach().cpu()
        data.detach().cpu()
        faceidx = np.where(y==1)[0]
        scrambleidx = np.where(y==0)[0]
        #vkllloss += vkllloss.item()
        #print(totalloss.item())
        fout = out[faceidx]
        sout = out[scrambleidx]
        fmu = mu[faceidx]
        fv = v[faceidx]
        smu = mu[scrambleidx]
        sv = v[scrambleidx]
        tb.add_image("Validation Face Data(decode)" , torch.logit(fout) , e*10+i  , dataformats='NCHW')
        tb.add_image("Validation Scramble(decode)", torch.logit(sout) , e*10+i  , dataformats='NCHW')
        tb.add_histogram(f'mu Histogram',mu, e*10+i)
        tb.add_histogram(f'Face mu Histogram',fmu, e*10+i)
        tb.add_histogram(f'Face v Histogram',fv, e*10+i)
        tb.add_histogram(f'Scramble mu Histogram',smu, e*10+i)
        tb.add_histogram(f'Scramble v Histogram',sv, e*10+i)
        tb.add_scalars("Face/Scramble Count", {'face':fmu.shape[0],'scramble':smu.shape[0]}, e*10+i)

    if(vloss<bestvloss):
        torch.save(coder,'./Model_CNS/bmodel_'+str(vloss/vbatch)+'_epoch_'+str(e))
        bestvloss=vloss
    tb.add_scalar("Training Avg Loss", (trainingloss/batch), e)
    tb.add_scalar("Training Avg kloss", (kloss/batch), e)
    tb.add_scalar("Validation Avg Loss", (vloss/vbatch), e)
    tb.add_scalar("Validation Avg kloss", (vkloss/vbatch), e)


    print(f'epoch: {e:3} batch: {i:3} Training Batch Avg loss: {trainingloss/batch:10.8f} Validation Batch Avg loss: {vloss/vbatch:10.8f}  Validation Avg R loss: {vrloss/vbatch:10.8f}, Validation Avg K loss {vkloss/vbatch:10.8f}')
torch.save(coder,'./Model_CNS/bmodel_Lastone')
