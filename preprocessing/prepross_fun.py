import numpy as np
from scipy.signal import butter, filtfilt, periodogram
from sklearn import preprocessing
from scipy.stats import entropy


def filter_data_bp(x,freqz, fs, filt_type='bandpass',order=5):
    """Filter data. Input: np.array, shape = (batch,seq_len,ch)"""
    b, a = butter(order, freqz, btype=filt_type,fs=fs)
    return filtfilt(b, a, x,axis=1)
    

def filter_data_hp(x,lowcut, fs, filt_type='highpass',order=5):
    """Filter data. Input: np.array, shape = (batch,seq_len,ch)"""
    b, a = butter(order, lowcut, btype=filt_type,fs=fs)
    return filtfilt(b, a, x,axis=1)


def scaler_scipy(x_train,x_val,mode=0):
    if mode==0:
        print("Normalize [0,1]")
        scale=preprocessing.MinMaxScaler(copy=False).fit(x_train.reshape(-1,x_train.shape[2]))

    if mode==1:
        print("Normalize [-1,1]")
        scale=preprocessing.MinMaxScaler(feature_range=(-1,1),copy=False).fit(x_train.reshape(-1,x_train.shape[2]))

    if mode==1:
        print("Standardize [0 mean, 1 std")
        scale=preprocessing.StandardScaler(copy=False).fit(x_train.reshape(-1,x_train.shape[2]))

    x_train = scale.transform(x_train.reshape(-1,x_train.shape[2])).reshape(x_train.shape)
    x_val = scale.transform(x_val.reshape(-1,x_val.shape[2])).reshape(x_val.shape)
    return x_train,x_val

#------------------ Extend number of training samples-----------#

def data_split_n_extend(x,y,split):
    """Splits the time series into windows, where each window is given the same lable as the original sample."""
    x_ext=x.reshape(int(x.shape[0]*split),int(x.shape[1]/split),x.shape[2])
    y_ext=np.empty((int(x.shape[0]*split)))

    for i in range(x.shape[0]):
        if y[i]==2:
            y_ext[i*split:(i+1)*split]=2
        elif y[i]==1:
            y_ext[i*split:(i+1)*split]=1
        else:
            y_ext[i*split:(i+1)*split]=0
    y_ext=y_ext.astype(int)
    x_ext=x_ext.astype(np.float32)
    print(x_ext.shape,x_ext.dtype,y_ext.shape,y_ext.dtype,y_ext.sum()/y_ext.shape[0]==y.sum()/y.shape[0])

    return x_ext,y_ext

def overlap_split(x,y):
    """Creates 50% overlapping 10 second segments"""
    split=5
    x_ext=np.zeros((x.shape[0]*split, int(x.shape[1]/3),x.shape[2])).astype(np.float32)
    y_ext=np.empty((int(x.shape[0]*split)))

    for i in range(x.shape[0]):
        split1=np.split(x[i,0:3200].astype(np.float32),split,axis=0)#10x3200x14
        split2=np.split(x[i,640:].astype(np.float32),split,axis=0)
        x_ext[i*split]  =np.concatenate((split1[0], split1[1]),axis=0).astype(np.float32)
        x_ext[i*split+1]=np.concatenate((split2[0], split2[1]),axis=0).astype(np.float32)
        x_ext[i*split+2]=np.concatenate((split1[2], split1[3]),axis=0).astype(np.float32)
        x_ext[i*split+3]=np.concatenate((split2[2], split2[3]),axis=0).astype(np.float32)
        x_ext[i*split+4]=np.concatenate((split1[4], split2[4]),axis=0).astype(np.float32)

        if y[i]==2:
            y_ext[i*split:(i+1)*split]=2
        elif y[i]==1:
            y_ext[i*split:(i+1)*split]=1
        else:
            y_ext[i*split:(i+1)*split]=0

    y_ext=y_ext.astype(int)
    print(x_ext.shape,x_ext.dtype,y_ext.shape,y_ext.dtype,y_ext.sum()/y_ext.shape[0]==y.sum()/y.shape[0])
    return x_ext,y_ext


def bp_n_cat(x):
    """Concatenates bandpassed versions of x along the channels. OBS: doesnt increase number of samples, only number of channels."""
    x=np.concatenate((filter_data_bp(x[:,128*3:,:],[1,4],fs=128).astype(np.float32),
                      filter_data_bp(x[:,128*3:,:],[4,8],fs=128).astype(np.float32),
                      filter_data_bp(x[:,128*3:,:],[8,13],fs=128).astype(np.float32),
                      filter_data_bp(x[:,128*3:,:],[13,30],fs=128).astype(np.float32),
                      filter_data_bp(x[:,128*3:,:],[30,50],fs=128).astype(np.float32)), axis=2)
    print(x.shape)
    return x


#------------------ Hand crafted features-----------------------#

def hjort_features(x):
    """Calculates temporal features and hjort features from the samples, 98 features per sample"""
    batch=x.shape[0]
    features=np.zeros((batch,14,7))
    N=x.shape[1]

    mean=x.mean(axis=1).reshape(batch,1,14)

    var=x.var(axis=1).reshape(batch,1,14)

    skewness=1/N*(np.sum(x-mean))**3/x.std(axis=1)**3

    kurtosis=1/N*(np.sum(x-mean))**4/x.std(axis=1)**4

    mean=mean.reshape(batch,14)
    var=var.reshape(batch,14)

    x_prim=np.gradient(x,1/128,axis=1)

    x_biss=np.gradient(x_prim,1/128,axis=1)

    mobility_x=np.sqrt(x_prim.var(axis=1)/var)

    mobility_x_prim=np.sqrt(x_biss.var(axis=1)/x_prim.var(axis=1))

    complexity=mobility_x_prim/mobility_x

    amplitude=np.abs(x.max(axis=1)-x.min(axis=1))

    features[:,:,0]=mean
    features[:,:,1]=var
    features[:,:,2]=skewness
    features[:,:,3]=kurtosis
    features[:,:,4]=mobility_x
    features[:,:,5]=complexity
    features[:,:,6]=amplitude

    features=features.reshape(batch,-1).astype(np.float32)
    print(features.shape,features.dtype)
    return features
    
def calc_DE(x):
    var=x.std(axis=1)**2
    return 1/2*np.log(2*np.e*np.pi*var)

def filter_n_DE(x):
    """Applies a butterworth bandpass filter on the segments, and then calculates
    the DE for 5 different frequency bands for each channel, i.e. 70 features per sample""" 
    fs=128
    bp0=filter_data_bp(x,[1,4],fs=128).astype(np.float32)
    bp1=filter_data_bp(x,[4,8],fs=128).astype(np.float32)
    bp2=filter_data_bp(x,[8,13],fs=128).astype(np.float32)
    bp3=filter_data_bp(x,[13,30],fs=128).astype(np.float32)
    bp4=filter_data_bp(x,[30,50],fs=128).astype(np.float32)
    
    feats=np.concatenate((calc_DE(bp0),calc_DE(bp1),calc_DE(bp2),calc_DE(bp3),calc_DE(bp4)),axis=1)
    print(feats.shape,feats.dtype)
    return feats


def spectral_features(x):
    """calculates specral features from five frequency bands, 224 features per sample"""
    fs=128
    n=x.shape[1]
    bp0=filter_data_bp(x,[1,4],fs).astype(np.float32)
    bp1=filter_data_bp(x,[4,8],fs).astype(np.float32)
    bp2=filter_data_bp(x,[8,13],fs).astype(np.float32)
    bp3=filter_data_bp(x,[13,30],fs).astype(np.float32)
    bp4=filter_data_bp(x,[30,50],fs).astype(np.float32)

    e0=np.sum(np.abs(bp0)**2,axis=1)
    e1=np.sum(np.abs(bp1)**2,axis=1)
    e2=np.sum(np.abs(bp2)**2,axis=1)
    e3=np.sum(np.abs(bp3)**2,axis=1)
    e4=np.sum(np.abs(bp4)**2,axis=1)

    total_e=e0+e1+e2+e3+e4
    frac_e0=e0/total_e
    frac_e1=e1/total_e
    frac_e2=e2/total_e
    frac_e3=e3/total_e
    frac_e4=e4/total_e

    se0=entropy(periodogram(bp0,fs=fs, axis=1)[1],axis=1)/np.log(n/2)
    se1=entropy(periodogram(bp1,fs=fs, axis=1)[1],axis=1)/np.log(n/2)
    se2=entropy(periodogram(bp2,fs=fs, axis=1)[1],axis=1)/np.log(n/2)
    se3=entropy(periodogram(bp3,fs=fs, axis=1)[1],axis=1)/np.log(n/2)
    se4=entropy(periodogram(bp4,fs=fs, axis=1)[1],axis=1)/np.log(n/2)

    feats=np.concatenate((e0,e1,e2,e3,e4,total_e,frac_e0,frac_e1,frac_e2,frac_e3,frac_e4,se0,se1,se2,se3,se4),axis=1)
    print(feats.shape, feats.dtype)
    return feats

