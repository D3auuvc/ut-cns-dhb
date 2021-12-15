#!/usr/bin/python3
import numpy as np
from torchsummary import summary
import pickle
import torch
import datetime
from VAE import VAE
## Training
bceloss=torch.nn.BCELoss()
#print(fftdata[1][111,0,0])
epoch=500
batchsize=10
coder = VAE(3,3)
optimizer = torch.optim.Adam(coder.parameters(), lr=0.001)
f=open('./FFT_AllSubject_Training_Face_minmaxNorm','rb')
face_FFTdata=pickle.load(f)
face_FFTresult = np.zeros((1,3,102,188))
for l in range(len(face_FFTdata)):
    face_FFTdata[l]=np.where(face_FFTdata[l]>1,1,face_FFTdata[l])
    #totalsample+SummaryWriterhape[0]
    face_FFTresult = np.vstack((face_FFTresult,face_FFTdata[l]))

#tb = SummaryWriter('Training_CNS/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'_R13_bsize10_shuffleOutput_weightInit_noKDL_RAW_500epoch')
length=[x for x in range(face_FFTresult.shape[0])]
t=int(np.floor(face_FFTresult.shape[0]*0.7))
np.random.shuffle(length)
print(t)
tidxs=length[:t]
vidxs=length[t:]
#print(tidxs)
batch=t//batchsize
vbatch=(face_FFTresult.shape[0]-t)//batchsize
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
        oidx=idx.copy()
        np.random.shuffle(oidx)
        #print(oidx)
        data = torch.tensor(face_FFTresult[idx,:,:,:]).float().to(coder.device)
        odata = torch.tensor(face_FFTresult[oidx,:,:,:]).float().to(coder.device)
        out = coder(data)
        regconstructionloss = bceloss(out,odata)
        #KLDloss = - 0.5 * torch.sum(1+ v - mu.pow(2) - v.exp())
            #print(f'R {regconstructionloss}')
            #print(f'K {KLDloss}')
        totalloss=regconstructionloss#+KLDloss
        optimizer.zero_grad()
        trainingloss+=totalloss.item()
        rloss+=regconstructionloss.item()
        #kloss+=KLDloss.item()
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
        #print(idx)
        data = torch.tensor(face_FFTresult[idx,:,:,:]).float().to(coder.device)
        out = coder(data)
        regconstructionloss = bceloss(out,data)
        #KLDloss = - 0.5 * torch.sum(1+ v - mu.pow(2) - v.exp())
            #print(f'R {regconstructionloss}')
            #print(f'K {KLDloss}')
        loss=regconstructionloss#+KLDloss
        vloss+=loss.item()
        vrloss+=regconstructionloss.item()
        #vkloss+=KLDloss.item()
        #print(totalloss.item())
    if(vloss<bestvloss):
        torch.save(coder,'./Model_CNS/bmodel_'+str(vloss/vbatch))
        bestvloss=vloss

    #mm = torch.median(mu)
    #mq25 = torch.quantile(mu,0.25)
    #mq75 = torch.quantile(mu,0.75)
    #mv = torch.median(v)
    #vq25 = torch.quantile(v,0.25)
    #vq75 = torch.quantile(v,0.75)
    #tb.add_scalars("Validation Mean", {'median':mm,'q25':mq25,'q75':mq75}, e)
    #tb.add_scalars("Validation V", {'median':mv,'q25':vq25,'q75':vq75}, e)

    print(f'epoch: {e:3} batch: {i:3} Training Batch Avg loss: {trainingloss/batch:10.8f} Validation Batch Avg loss: {vloss/vbatch:10.8f}  Validation Avg K loss: {vkloss/vbatch:10.8f},  Validation Avg R loss: {vrloss/vbatch:10.8f}')
