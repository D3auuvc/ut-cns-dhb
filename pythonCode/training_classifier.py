#!/usr/bin/python3
import numpy as np
from torchsummary import summary
import pickle
import torch
import datetime
from torch.utils.tensorboard import SummaryWriter
from VAE_Model.DNN import DNN
## Training

def loadFeatureWithZ(path):
    f=open(path,'rb')
    feature=[]
    dim=102
    trainFeature=pickle.load(f)
    for i in range(len(trainFeature['face']['z'])):
        for j in  range(trainFeature['face']['z'][i].shape[0]):
            fv=np.zeros(dim+1)
            fv[:dim]= trainFeature['face']['z'][i][j]
            #fv[2048:4096]= trainFeature['face']['v'][i][j].numpy()
            fv[dim]=1
            feature.append(fv)

    for i in range(len(trainFeature['scramble']['z'])):
        for j in  range(trainFeature['scramble']['z'][i].shape[0]):
            fv=np.zeros(dim+1)
            fv[:dim]= trainFeature['scramble']['z'][i][j]
            #fv[2048:4096]= trainFeature['scramble']['v'][i][j].numpy()
            fv[dim]=0
            feature.append(fv)
    return feature


tb = SummaryWriter('Training_CNS_TensorBoard/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'_R108_bsize50_ksize4_Avgpool_5e6d_FFTVSFFT_noTransform_AllClass_flattenStackFeatureColumn102_reduceMnet_noinitWeight_withKLDloss')
feature = loadFeatureWithZ('./Model_output_Feature/Training_Feature_R95E437_102_AllClass_kKLDloss_withZ')
feature = np.array(feature)
config = [102,200,100,50,1]
classifer = DNN(config)
print(classifer)
loss=torch.nn.BCEWithLogitsLoss()
#print(fftdata[1][111,0,0])
epoch=500
batchsize=50
optimizer = torch.optim.Adam(classifer.parameters(), lr=0.001)
length=[x for x in range(feature.shape[0])]
t=int(np.floor(feature.shape[0]*0.7))
np.random.shuffle(length)
print(t)
tidxs=length[:t]
vidxs=length[t:]
batch=t//batchsize
vbatch = (feature.shape[0]-t)//batchsize
bestvloss=float('inf')
for e in range(epoch):
    trainingloss=0
    vloss=0
    classifer.train()
    for i in range(batch):
        idx=tidxs[i*batchsize:i*batchsize+batchsize]
        #oidx=idx.copy()
        #np.random.shuffle(oidx)
        #print(oidx)
        #data = torch.tensor(raw_result[idx,:,:,:]).float().to(coder.device)
        data = torch.tensor(feature[idx, 0:102]).float().to(classifer.device)
        y = torch.tensor(feature[idx, 102:103]).float().to(classifer.device)
        yhat = classifer(data)
        #print(torch.max(v))
        l  = loss(yhat,y)
        trainingloss += l.item()
        #sKDloss = -0.5 * torch.mean( torch.sum(1+ sv - smu.pow(2) - sv.exp()))
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

    classifer.eval()
    accuracy = 0
    for i in range(vbatch):
        idx=vidxs[i*batchsize:i*batchsize+batchsize]
        #oidx=idx.copy()
        #np.random.shuffle(oidx)
        #print(idx)
        #data = torch.tensor(raw_result[idx,:,:,:]).float().to(coder.device)
        data = torch.tensor(feature[idx,0:102]).float().to(classifer.device)
        y = torch.tensor(feature[idx,102:103]).float().to(classifer.device)
        yhat = classifer(data)
        vl = loss(yhat,y)
        y=y.detach().cpu().numpy()
        yhat=yhat.detach().cpu().numpy()
        #sKDloss = -0.5 * torch.mean( torch.sum(1+ sv - smu.pow(2) - sv.exp()))
        vloss+=vl.item()
        faceidx = np.where(y==1)[0]
        scrambleidx = np.where(y==0)[0]
        #vkllloss += vkllloss.item()
        #print(totalloss.item())
        result=np.where(yhat==y)
        accuracy+=result[0].shape[0]

    if(e%10==0):
        print(f'Sample epoch: {e:3} batch: {i:3} Validation predict: {yhat.reshape(1,-1)}, target  {y.reshape(1,-1)}')
    if(vloss<bestvloss):
        torch.save(classifer,'./Model_CNS/Classifier_bmodel_'+str(vloss/vbatch)+'_epoch_'+str(e))
        bestvloss=vloss
    tb.add_scalar("Training Avg Loss", (trainingloss/batch), e)
    tb.add_scalar("Validation Avg Loss", (vloss/vbatch), e)
    tb.add_scalar("Validation Accuracy", (accuracy/vbatch), e)

    print(f'epoch: {e:3} batch: {i:3} Training Batch Avg loss: {trainingloss/batch:10.8f}, Validation Batch Avg loss: {vloss/vbatch:10.8f}, Validation Accuracy: {accuracy/vbatch}')
torch.save(classifer,'./Model_CNS/Classifier_bmodel_Lastone')
