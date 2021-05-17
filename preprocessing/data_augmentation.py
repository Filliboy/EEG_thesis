import numpy as np

def heart_beat_augmentation_off(samples):
    '''Adds an artificial heartbeat in random channels (30% chance) of each sample'''
    N=samples.shape[1]
    fs=128
    T=1/fs
    freq= (0.9 - 1.4) * np.random.rand(1) + 1.4
    beat_range = np.linspace(0, N*T, N)

    ch_ind=np.random.rand(14)>0.7
    samples[:,:,ch_ind]=samples[:,:,ch_ind]+20*np.sin(freq*2*np.pi*beat_range).reshape(1,-1,1)
    return samples.astype(np.float32)

def eye_blink_augmentation_off(samples):
    '''Adds two artificial eye blinks in the top and bottom four channels (forehead)'''
    N=50
    fs=128
    T=1/fs
    sig_len=samples.shape[1]
    beat_range = np.linspace(0, N*T, N)

    ch_ind=list(range(4))+list(range(10,14))
    aug_ind_low=np.random.randint(sig_len/2-50)
    aug_ind_high=np.random.randint(sig_len/2,sig_len-50)

    samples[:,aug_ind_low:aug_ind_low+N,ch_ind]=samples[:,aug_ind_low:aug_ind_low+N,ch_ind]+100*np.sin(1*2*np.pi*beat_range).reshape(1,-1,1)
    samples[:,aug_ind_high:aug_ind_high+N,ch_ind]=samples[:,aug_ind_high:aug_ind_high+N,ch_ind]+100*np.sin(1*2*np.pi*beat_range).reshape(1,-1,1)
    return samples.astype(np.float32)

def lost_connection_augmentation_off(samples):
    '''Adds one large pertupation in a random channel in each sample'''
    N=50
    fs=128
    T=1/fs
    per_range = np.linspace(0, N*T, N)
    
    for i in range(samples.shape[0]):
        ch_ind=np.random.rand(14).argmax()
        aug_ind=np.random.randint(samples.shape[1]-N)
        freq= (0.8 - 1.2) * np.random.rand(1) + 1.2
        hight= (400 - 700) * np.random.rand(1) + 700
        samples[i,aug_ind:aug_ind+N,ch_ind]=samples[i,aug_ind:aug_ind+N,ch_ind] + hight*np.cos(freq*2*np.pi*per_range)
    return samples.astype(np.float32)

def flip_augmentation_off(samples):
    """flips the segment"""
    samples=np.flip(samples,1)
    return samples.astype(np.float32)

def random_segment_augmentation_off(samples):
    """create a new segment from three short segment (works only when split into three's)"""
    bs=samples.shape[0]
    length=samples.shape[1]
    if bs%3==0:
        times=int(bs/3)
    else:
        times=int((bs-bs%3)/3)
    new_samples=np.empty(samples.shape)

    for i in range(times):
        seg=np.concatenate((samples[i*3,:,:],samples[i*3+1,:,:],samples[i*3+2,:,:]),axis=0)
        for j in range(3):
            samp_ind=np.random.randint(seg.shape[0]-length)
            new_samples[i*3+j]=seg[samp_ind:samp_ind+length]

    return new_samples.astype(np.float32)

def augmentation_pipeline_offline(samples,labels,bs):
    n_samples=samples.shape[0]
    ind=np.arange(n_samples)
    augmentation_list=np.array((heart_beat_augmentation_off,
                       eye_blink_augmentation_off,
                       lost_connection_augmentation_off,
                       flip_augmentation_off,
                       random_segment_augmentation_off))

    num_augments=augmentation_list.shape[0]
    
    for i,batch in enumerate(range(0,n_samples,bs)):
        augs=[]

        batch_ind=ind[batch:batch+bs]
        
        if i==0:
           x=samples[batch_ind]
           y=labels[batch_ind]
           
           for _ in range(num_augments):
               aug_inds=np.random.rand(num_augments)
               if (aug_inds>0.5).any():
                  temp=x.copy()
                  for augmentation in augmentation_list[aug_inds>0.5]:
                      temp=augmentation(temp)
               else:
                  temp=augmentation_list[aug_inds.argmax()](x)
               augs.append(temp)

           x=np.concatenate((x, augs[0], augs[1],augs[2],augs[3],augs[4]))
           y=np.concatenate((y,y,y,y,y,y))
        else:
            x_batch=samples[batch_ind]
            y_batch=labels[batch_ind]

            for i in range(num_augments):
                aug_inds=np.random.rand(num_augments)
                if (aug_inds>0.5).any():
                   temp=x_batch.copy()
                   for augmentation in augmentation_list[aug_inds>0.5]:
                       temp=augmentation(temp)
                else:
                   temp=augmentation_list[aug_inds.argmax()](x_batch)

                augs.append(temp)
            x=np.concatenate((x,x_batch, augs[0], augs[1],augs[2],augs[3],augs[4]))
            y=np.concatenate((y,y_batch,y_batch,y_batch,y_batch,y_batch,y_batch))
    print(x.shape,x.dtype,y.shape,y.dtype)
    return x,y