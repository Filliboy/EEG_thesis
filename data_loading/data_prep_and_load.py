import pickle
import re
import pandas as pd
import numpy as np
import os

def natural_sort_key(s):
    """Sorts a list in rising order after the number""" 
    _nsre = re.compile('([0-9]+)')

    return [int(text) if text.isdigit() else text.lower()
          for text in re.split(_nsre, s)]

def by_letter(elem):
    if elem[0][0:2].isdigit():
        sort=elem[0][3:]
    else:
        sort=elem[0][2:]
    return sort

def extract_n_sort_samples(csv_dir,ratings_dir):
    """Input: directory to csv file and corresponding rating. Output: dict of numpy array with samples extracted from the dataframe
    and corresponding labels sorted after song."""

    data=pd.read_csv(csv_dir, header=1)

    channels=data[["EEG.AF3", "EEG.F7", "EEG.F3", "EEG.FC5", "EEG.T7", "EEG.P7", "EEG.O1", "EEG.O2", "EEG.P8", "EEG.T8", "EEG.FC6", "EEG.F4", "EEG.F8", "EEG.AF4"]].to_numpy(dtype=np.float32)
    mark_int=data['MarkerValueInt']

    start_ind=mark_int[mark_int==21].index.to_numpy()
    fail_ind=mark_int[mark_int==23].index.to_numpy()

    # start_ind=start_ind[1:]
    # fail_ind=fail_ind[2:]
    if fail_ind.shape[0] != 0:
        for fail in fail_ind:
            #delete failed samples
            ind=np.where(start_ind==start_ind[start_ind<fail].max())
            start_ind=np.delete(start_ind,ind)
            
    print(start_ind)
    eeg_samples=np.empty((start_ind.shape[0], int(128*33), 14), dtype=np.float32)
    for i,start in enumerate(start_ind):
        eeg_samples[i]=channels[start:start+128*33,:]

    with open(ratings_dir, 'rb') as f:
        data= pickle.load(f)

    song,like,val,ar,fam,samps=zip(*sorted(zip(data['Song'],data['Likeability'],data['Valance'],data['Arousal'],data['Familiarity'],eeg_samples),key=by_letter))

    x=np.array(samps)
    y=np.array((like,val,ar,fam))
    print(x.shape,y.shape)
    for s in song:
        print(s)
    return {'x':x,'y':y}


def dump_pick(data_dict,path, test_id):
    f = open(path+str(test_id)+'.pkl',"wb")
    pickle.dump(data_dict,f)
    f.close()


def load_pkl(dirr):
    """Open pickled deap dict. Label order for DEAP: valence, arousal, dominance, liking.
    For collected data: Liking, valence, arousal, familiarity"""
    with open(dirr, 'rb') as f:
      data= pickle.load(f)
    return data

def load_d(path):
    """Load the data to memory from pickled samples"""
    data_dir=os.listdir(path)
    data_dir.sort(key=natural_sort_key)

    for i,test in enumerate(data_dir):
        data=load_pkl(path+test)
        if i==0:
            x=data['x']
            y=data['y'].T
        else:
            temp_x=data['x']
            temp_y=data['y'].T
            x=np.concatenate((x,temp_x))
            y=np.concatenate((y,temp_y))
    print(x.shape,x.dtype,y.shape,y.dtype)
    return x,y


def split_labels_per_subject(path,label=0):
    '''For three class tasks. Normalize ratings between 0 (min) and 1 (max) and split into three classes.
    Label=0 likeability, 1=valence,2=arousal'''
    data_dir=os.listdir(path)
    data_dir.sort(key=natural_sort_key)

    for i,test in enumerate(data_dir):
        data=load_pkl(path+test)
        if i==0:
            y=data['y'][label,:]
            y=(y-y.min())/(y.max()-y.min())
            y[y<0.33]=1
            y[y<0.66]=2
            y[y<1]=3
            y=y-1

        else:
            temp_y=data['y'][label,:]
            temp_y=(temp_y-temp_y.min())/(temp_y.max()-temp_y.min())
            temp_y[temp_y<0.33]=1
            temp_y[temp_y<0.66]=2
            temp_y[temp_y<1]=3
            temp_y=temp_y-1
            y=np.concatenate((y,temp_y))
    y=y.astype(int)

    if label==0:
      print('Likeability')
    if label==1:
      print('Valence')
    if label==2:
      print('Arousal')
    print(y.shape,y.dtype)
    return y

def bin_label(y,lable):
    """Returns binary labels. Label=0 likeability, 1=valence,2=arousal"""
    return (y[:,lable]>5).astype(int)

