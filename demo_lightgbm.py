from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
from preprocessing import prepross_fun
import lightgbm as lgb
import numpy as np
from data_loading.data_prep_and_load import load_d, bin_label


def main():

    path="data_demo/"

    x,y=load_d(path)

    y=bin_label(y,label=1)

    feats=prepross_fun.filter_n_DE(x[:,128*3:,:])

    kf= KFold(5,shuffle=False)

    param = { 'objective':'binary','max_bin': 512,'verbose': -1,'first_metric_only': True}
    param['metric'] = ['binary_error','auc','binary_logloss']
    num_round = 1000
    preds=np.zeros((30,5))
    aucs=np.zeros((30,5))

    for j in range(2,30):
        param['num_leaves']=j
        
        for i,(train_index,test_index) in enumerate(kf.split(feats)):
                scaler=preprocessing.MinMaxScaler().fit(feats[train_index])

                x_train=scaler.transform(feats[train_index])
                x_test=scaler.transform(feats[test_index])

                train_data=lgb.Dataset(x_train,label=y[train_index])
                valid_data=lgb.Dataset(x_test,label=y[test_index])

                bst = lgb.train(param, train_data, num_round, valid_sets=[valid_data], early_stopping_rounds=10, verbose_eval=False)
                # bst.save_model('model'+str(i)+'.txt', num_iteration=bst.best_iteration)
                
                ypred = bst.predict(x_test)
                preds[j-2,i]=((ypred>0.5).astype(int)==y[test_index]).sum()/y[test_index].shape[0]   
                aucs[j-2,i]=roc_auc_score(y[test_index],ypred)

        print("Mean acc: "+ str(preds[j-2].mean()),"Mean AUC: "+ str(aucs[j-2].mean()),"Num leaves: " +str(j))
    print("Best round acc: "+str(preds.mean(axis=1).max()),
    "Best round AUC: "+ str(aucs.mean(axis=1)[preds.mean(axis=1).argmax()]),
     "Num leaves: "+str(preds.mean(axis=1).argmax()+2))

if __name__ =='__main__':
    main()