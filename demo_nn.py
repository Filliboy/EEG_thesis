
from data_loading import data_prep_and_load
from preprocessing import prepross_fun
from training_n_evaluation.train_n_eval import train_kfold, majority_voting  
import numpy as np
from sklearn.model_selection import KFold
from models import models


def main():
    path="C:/Users/filip/OneDrive/Exjobb/data_collection/data_ordered/"

    x,y=data_prep_and_load.load_d(path)
    
    x = prepross_fun.bp_n_cat(x)
    #Likeability
    y = data_prep_and_load.bin_label(y,lable=0)
    
    kf= KFold(5,shuffle=False)
 
    gen= kf.split(x)
    indices=list(zip(*gen))
    no_class=1

    all_samp_acc=np.zeros(5)
    all_maj_acc=np.zeros(5)
    for i in range(5):
        model=models.EEG_net(seq_len=1280,in_ch=70, bottle_neck_size=20,filters=32, num_classes=no_class)

        model,x_val,y_val= train_kfold(i,indices,x,y,model,ss=0,os=1,aug=0)

        maj_acc,samp_acc=majority_voting(model,x_val,y_val,num_classes=no_class,split=5)

        print(maj_acc,samp_acc,i)
        all_maj_acc[i]=maj_acc
        all_samp_acc[i]=samp_acc

    print(all_maj_acc.mean(),all_samp_acc.mean())

if __name__ =='__main__':
    main()