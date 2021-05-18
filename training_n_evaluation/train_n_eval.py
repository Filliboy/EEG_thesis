from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl
from preprocessing import prepross_fun
from preprocessing import data_augmentation
import numpy as np
import torch as t
import torchmetrics

def basic_trainer(train_d,test_d,model,bs,epochs,early_stopping=0):
    train_loader=t.utils.data.DataLoader(train_d,batch_size=bs,shuffle=True)
    val_loader=t.utils.data.DataLoader(test_d,batch_size=bs)

    if early_stopping==0:
        trainer= pl.Trainer(gpus=1,progress_bar_refresh_rate=100,max_epochs=epochs)
    else:
        early_stop_callback = EarlyStopping(monitor='Validation accuracy', min_delta=0.00, patience=3, verbose=False, mode='max')
        trainer= pl.Trainer(callbacks=[early_stop_callback], gpus=1, progress_bar_refresh_rate=0, max_epochs=epochs,min_epochs=20)

    trainer.fit(model, train_loader,val_loader)

    return model

def train_kfold(i, indices,x,y,model,ss=1,os=0,aug=0,bs=32,epochs=100):
    """Can be used in a loop, where "i" is the current fold"""
    train_index= indices[0][i]
    test_index = indices[1][i]

    if ss==1:
        x_ext_train,y_ext_train=prepross_fun.data_split_n_extend(x[train_index],y[train_index], 3)
        x_ext_val,y_ext_val=prepross_fun.data_split_n_extend(x[test_index],y[test_index],3)

    if os==1:
        x_ext_train,y_ext_train=prepross_fun.overlap_split(x[train_index],y[train_index])

        x_ext_val,y_ext_val=prepross_fun.overlap_split(x[test_index],y[test_index])

    if aug==1:
        x_ext_train,y_ext_train=data_augmentation.augmentation_pipeline_offline(x_ext_train,y_ext_train,bs=9)

    x_train,x_val=prepross_fun.scaler_scipy(x_ext_train,x_ext_val)

    x_train=t.tensor(x_train)
    x_val=t.tensor(x_val)

    x_train=x_train.view(x_train.shape[0],x_train.shape[2],x_train.shape[1])
    x_val=x_val.view(x_val.shape[0],x_val.shape[2],x_val.shape[1])

    y_train=t.tensor(y_ext_train)
    y_val=t.tensor(y_ext_val)

    train_d=t.utils.data.TensorDataset(x_train,y_train)
    test_d=t.utils.data.TensorDataset(x_val,y_val)

    model=basic_trainer(train_d, test_d, model, bs=bs,epochs=epochs, early_stopping=1)

    return model,x_val,y_val

def majority_voting(model,x,y,num_classes,split):
    """Gives the sample the majority prediction of the shorter samples"""
    size=int(x.shape[0]/split)
    with t.no_grad():
      x= x.to(t.device("cuda:0"))
      model=model.to(t.device("cuda:0"))
      y_hat=model(x) 
      y_hat=y_hat.to(t.device("cpu"))
    maj_y=t.empty(size)

    max_list=np.zeros(5)
    h=0
    for i in range (size):
        if num_classes==1:
           for j in range(split):
               max_list[j]=(y_hat[i*split+j]>0.5).to(int)

           if max_list.sum()>=(split//2+1):
              maj_y[i]=1
           else:
              maj_y[i]=0
        else:
            for j in range(split):
                max_list[j]= y_hat[i*split+j].argmax()

            hist=np.histogram(max_list,3)[0]
            inds=np.argwhere(hist==hist.max())
            if inds.shape[0]==1:
               maj_y[i]=hist.argmax()
            else:
              #if vote is inconclusive, pick most confident vote
               for j in range(split):
                   max_list[j]= y_hat[i*split+j].max()

               max_seg=max_list.argmax()
               maj_y[i]=y_hat[i*split+max_seg].argmax()
               h+=1
        
    y_short=y[0::split]

    maj_acc=(maj_y==y_short).sum()/y_short.shape[0]
    print("Majority vote accuracy: " + str(maj_acc),h)

    if num_classes>1:
       samp_acc=torchmetrics.functional.accuracy(y_hat,y)
    else:
       samp_acc=torchmetrics.functional.accuracy(y_hat.view(-1),y)

    print("Class distribution: ",(y==0).sum()/y.shape[0],(y==1).sum()/y.shape[0],(y==2).sum()/y.shape[0])
    print("Per sample accuracy: "+ str(samp_acc))
    return maj_acc,samp_acc
