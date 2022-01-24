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


def reparametrize(mu,log_var):
    #Reparametrization Trick to allow gradients to backpropagate from the
    #stochastic part of the model
    sigma = torch.exp(0.5*log_var)
    z = torch.randn_like(log_var)
    #z= z.type_as(mu)
    return mu + sigma*z

X = loadData('./RawData/FFT_AllSubject_Training_AllClass_minmaxNorm',188)
Y = loadData('./RawData/FFT_AllSubject_Training_AllClass_Target',1)
coder = torch.load('./Model_CNS/R108/bmodel_0.12110783931400095_epoch_80')

loss = torch.nn.BCEWithLogitsLoss()
tb = SummaryWriter('Testing_CNS/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'_R108E80_3ZSample')
idx=[x for x in range(X.shape[0])]
np.random.shuffle(idx)
batchsize=50
batch=X.shape[0]//batchsize
vloss=0
data = np.zeros((20,3,102,188))
for i in range(20):
    data[i]=np.random.rand(3,102,188)

for i in range(1):
    bvloss=0
    coder.eval()
    #bidx=idx[i*batchsize:i*batchsize+batchsize]
    #data = torch.tensor(result[bidx,:,:,0:102]).float().to(coder.device)
    data = torch.tensor(data).float().to(coder.device)
    #fdata = torch.tensor(fft_result[bidx,:,:,0:102]).float().to(coder.device)
    out,mu,v = coder(data)
    out=out.detach().cpu()
    mu=mu.detach().cpu()
    v=v.detach().cpu()
    #regconstructionloss = loss(out,fdata)
    #vloss+=regconstructionloss.item()
    #bvloss+=regconstructionloss.item()
    tb.add_image("Random Data(decode) batch_"+str(i) , torch.logit(out) , i , dataformats='NCHW')
    tb.add_image("Random Data batch_"+str(i) , data , i , dataformats='NCHW')
    mu = torch.median(mu)
    mq25 = torch.quantile(mu,0.25)
    mq75 = torch.quantile(mu,0.75)
    mv = torch.median(v)
    vq25 = torch.quantile(v,0.25)
    vq75 = torch.quantile(v,0.75)
    tb.add_scalars("Random mu Statistic", {'median' : mu,'q25' : mq25,'q75' : mq75}, i)
    tb.add_scalars("Random v Statistic", {'median' : mv,'q25' : vq25,'q75' : vq75}, i)
    tb.add_scalar("Random Avg batch Loss", (bvloss/batchsize), i)

feature={ 'face':{'mean':[],'v':[] ,'z':[]} , 'scramble':{'mean':[],'v':[],'z':[]} }

for i in range(batch):
    bvloss=0
    coder.eval()
    bidx=idx[i*batchsize:i*batchsize+batchsize]
    data = torch.tensor(X[bidx]).float().to(coder.device)
    y = Y[bidx]
    #fdata = torch.tensor(fft_result[bidx,:,:,0:102]).float().to(coder.device)
    out,mu,v = coder(data)
    out=out.detach().cpu()
    mu=mu.detach().cpu()
    v=v.detach().cpu()
    faceidx = np.where(y==1)
    scrambleidx = np.where(y==0)
    fsize = faceidx[0].shape[0]
    ssize = scrambleidx[0].shape[0]
    # N Z Sample
    N =8
    fsample=np.zeros((fsize*N,102))
    ssample=np.zeros((ssize*N,102))
    for j in range(N):
        fsample[j*fsize:(j*fsize)+fsize]=reparametrize(mu[faceidx],v[faceidx]).numpy()
    for j in range(N):
        ssample[j*ssize:(j*ssize)+ssize]=reparametrize(mu[scrambleidx],v[scrambleidx]).numpy()
    # 1 Z sample
    #fsample = reparametrize(mu[faceidx],v[faceidx])
    #ssample = reparametrize(mu[scrambleidx],v[scrambleidx])
    fmu = torch.median(mu[faceidx])
    fmq25 = torch.quantile(mu[faceidx],0.25)
    fmq75 = torch.quantile(mu[faceidx],0.75)
    smu = torch.median(mu[scrambleidx])
    smq25 = torch.quantile(mu[scrambleidx],0.25)
    smq75 = torch.quantile(mu[scrambleidx],0.75)
    tb.add_scalars("Face mu Statistic", {'median' : fmu ,'q25' : fmq25,'q75' : fmq75}, i)
    tb.add_scalars("Scramble mu Statistic", {'median' : smu,'q25' : smq25,'q75' : smq75}, i)
    tb.add_image("Face Data(decode)" , torch.logit(out[faceidx]) , i  , dataformats='NCHW')
    tb.add_image("Scramble Data(decode)" , torch.logit(out[scrambleidx]) , i  , dataformats='NCHW')
    tb.add_histogram("Mean Histogram", mu, i)
    tb.add_histogram("Face Mean Histogram", mu[faceidx], i)
    tb.add_histogram("Face V Histogram",  v[faceidx], i)
    tb.add_histogram("Scramble Mean Histogram", mu[scrambleidx], i)
    tb.add_histogram("Scramble V Histogram",  v[scrambleidx], i)
    feature['face']['mean'].append(mu[faceidx])
    feature['face']['v'].append(v[faceidx])
    feature['face']['z'].append(fsample)
    feature['scramble']['mean'].append(mu[scrambleidx])
    feature['scramble']['v'].append(v[scrambleidx])
    feature['scramble']['z'].append(ssample)

f = open('./Training_Feature_R108E80_102_AllClass_kKLDloss_with8Z','wb')
pickle.dump(feature,f)


#print(f'Testing Avg loss: {vloss/batch:10.8f}')
