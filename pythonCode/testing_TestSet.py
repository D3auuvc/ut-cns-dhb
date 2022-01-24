#!/usr/bin/python3
import numpy as np
from torchsummary import summary
import pickle
import torch
import datetime
from torch.utils.tensorboard import SummaryWriter
#from VAE_RawVsFFT import VAE
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

X = loadData('./FFT_Test_minmaxNorm',188)
#Y = loadData('./FFT_AllSubject_Training_AllClass_Target',1)
#fft_result = loadData('./FFT_AllSubject_Training_Face_minmaxNorm',188)
coder = torch.load('./Model_CNS/R88/bmodel_0.11590400021523237_epoch_50')
loss = torch.nn.BCEWithLogitsLoss()
tb = SummaryWriter('Testing_CNS/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'_R23_TrainingR88E50_K4_f2048_TestData')
idx=[x for x in range(X.shape[0])]
batchsize=50
batch=X.shape[0]//batchsize
vloss=0
#data = np.zeros((20,3,102,188))
#for i in range(20):
#    data[i]=np.random.rand(3,102,188)

#for i in range(1):
#    bvloss=0
#    coder.eval()
#    bidx=idx[i*batchsize:i*batchsize+batchsize]
#    #data = torch.tensor(result[bidx,:,:,0:102]).float().to(coder.device)
#    data = torch.tensor(data).float().to(coder.device)
#    #fdata = torch.tensor(fft_result[bidx,:,:,0:102]).float().to(coder.device)
#    out,mu,v = coder(data)
#    #regconstructionloss = loss(out,fdata)
#    #vloss+=regconstructionloss.item()
#    #bvloss+=regconstructionloss.item()
#    tb.add_image("Random Data(decode) batch_"+str(i) , torch.logit(out) , i , dataformats='NCHW')
#    tb.add_image("Random Data batch_"+str(i) , data , i , dataformats='NCHW')
#    mu = torch.median(mu)
#    mq25 = torch.quantile(mu,0.25)
#    mq75 = torch.quantile(mu,0.75)
#    mv = torch.median(mu)
#    vq25 = torch.quantile(v,0.25)
#    vq75 = torch.quantile(v,0.75)
#    tb.add_scalars("Random mu Statistic", {'median' : mu,'q25' : mq25,'q75' : mq75}, i)
#    tb.add_scalars("Random v Statistic", {'median' : mv,'q25' : vq25,'q75' : vq75}, i)
#    tb.add_scalar("Random Avg batch Loss", (bvloss/batchsize), i)

feature={'mean':[],'v':[]}

for i in range(batch):
    bvloss=0
    coder.eval()
    #bidx=idx[i*batchsize:i*batchsize+batchsize]
    data = torch.tensor(X[i*batchsize:i*batchsize+batchsize,:,:,:]).float().to(coder.device)
    #fdata = torch.tensor(fft_result[bidx,:,:,0:102]).float().to(coder.device)
    out,mu,v = coder(data)
    out=out.detach().cpu()
    mu=mu.detach().cpu()
    v=v.detach().cpu()
    medianmu = torch.median(mu)
    mq25 = torch.quantile(mu,0.25)
    mq75 = torch.quantile(mu,0.75)
    tb.add_scalars("mu Statistic", {'median' : medianmu,'q25' : mq25,'q75' : mq75}, i)
    tb.add_image("FFT img (decode)" , torch.logit(out) , i  , dataformats='NCHW')
    tb.add_histogram("Mean Histogram", mu, i)
    tb.add_histogram("V Histogram",  v, i)
    feature['mean'].append(mu)
    feature['v'].append(v)

f = open('./Test_Feature_R88E50','wb')
pickle.dump(feature,f)


#print(f'Testing Avg loss: {vloss/batch:10.8f}')
