import torch as t
import pytorch_lightning as pl
import torchmetrics

#------------ResNet1D------------------#

class BasicBlock(t.nn.Module):

    def __init__(self, in_ch, filters,strides,do_flag):
        super(BasicBlock,self).__init__()
        
        self.conv1=t.nn.Conv1d(in_channels=in_ch, out_channels=filters,kernel_size= 3, stride=strides,padding=1)

        self.bn1=t.nn.BatchNorm1d(filters)

        self.relu1=t.nn.ReLU() 
        if do_flag:
          self.dropout1=t.nn.Dropout(0.2)
          self.dropout2=t.nn.Dropout(0.2)

        self.conv2=t.nn.Conv1d(in_channels=filters, out_channels=filters,kernel_size= 3, stride=1,padding=1)

        self.bn2=t.nn.BatchNorm1d(filters) 

        if strides !=1:
            self.convdown=t.nn.Conv1d(in_channels=in_ch,out_channels=filters,kernel_size=1, stride=strides)

            self.bndown=t.nn.BatchNorm1d(filters)
        
        self.relu2=t.nn.ReLU()
        
        self.strides=strides
        self.do_flag=do_flag

    def forward(self,x):
        x1=self.conv1(x)
        x1=self.bn1(x1)
        x1=self.relu1(x1)
        if self.do_flag:
          x1=self.dropout1(x1)
        x1=self.conv2(x1)
        x1=self.bn2(x1)

        if self.strides !=1:
            x2=self.convdown(x)
            x2=self.bndown(x2)
        else:
            x2=x

        x=x1+x2

        x=self.relu2(x)
        if self.do_flag:
          x=self.dropout2(x)

        return x

class ResNet(pl.LightningModule):

    def __init__(self, in_ch,seq_length,layer_blocks,do_flag,num_classes):
        super(ResNet,self).__init__()

        self.do_flag=do_flag
        self.num_classes=num_classes

        self.bn0=t.nn.BatchNorm1d(14, affine=False)
        self.conv1 = t.nn.Conv1d(in_channels=in_ch, out_channels=64, kernel_size=7, stride=2, padding=3)   
        self.bn1 = t.nn.BatchNorm1d(64) 
        self.relu=t.nn.ReLU(inplace=True)
        if do_flag:
          self.dropout=t.nn.Dropout(0.2)
        self.maxpool=t.nn.MaxPool1d(kernel_size=3,stride=2,padding=1)

        self.blocks=t.nn.ModuleList()
        self.blocks.append(BasicBlock(in_ch=64, filters=64, strides=1,do_flag=do_flag))

        for _ in range(layer_blocks[0] - 1):
            self.blocks.append(BasicBlock(64,64,strides=1,do_flag=do_flag))

        self.blocks.append(BasicBlock(64,128,strides=2,do_flag=do_flag))
        for _ in range(layer_blocks[1]-1):
            self.blocks.append(BasicBlock(128,128,strides=1,do_flag=do_flag))

        self.blocks.append(BasicBlock(128, 256,strides=2,do_flag=do_flag))
        for _ in range(layer_blocks[2]-1):
            self.blocks.append(BasicBlock(256,256,strides=1,do_flag=do_flag))

        self.blocks.append(BasicBlock(256, 512,strides=2,do_flag=do_flag))
        for _ in range(layer_blocks[3]-1):
            self.blocks.append(BasicBlock(512, 512,strides=1,do_flag=do_flag))

        self.avgpool=t.nn.AvgPool1d(kernel_size=int(seq_length/32),stride=1,padding=0)
        
        self.fc=t.nn.Linear(512,num_classes)

        if num_classes>1: 
          self.out_a=t.nn.Softmax(dim=1)
        else:
          self.out_a=t.nn.Sigmoid()

        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()

        # self.train_AUROC = pl.metrics.AUROC(num_classes=3)
        # self.valid_AUROC = pl.metrics.AUROC(num_classes=3)
        
    def forward(self, x):
        if self.aug:
           x = self.bn0(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.do_flag:
          x = self.dropout(x)
        x = self.maxpool(x)
        for block in self.blocks:
            x=block(x)
        x=self.avgpool(x)
        x=x.view(-1,512)
        x=self.fc(x)
        x=self.out_a(x)
        return x

    def training_step(self, train_batch):
        x,y=train_batch
        y_hat=self.forward(x)

        if self.num_classes>1:
          loss=t.nn.functional.cross_entropy(y_hat,y)
          acc=self.train_acc(y_hat,y)
        else:
          loss=t.nn.functional.binary_cross_entropy(y_hat.view(-1),y.to(t.float))
          acc=self.train_acc(y_hat.view(-1),y)

        # auroc=self.train_AUROC(y_hat,y)
        # auroc.detach()

        self.log('Train accuracy',acc)
        # self.log('Train AUROC',auroc)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x,y=val_batch
        y_hat=self.forward(x)

        if self.num_classes>1:
          loss=t.nn.functional.cross_entropy(y_hat,y)
          acc=self.valid_acc(y_hat,y)
        else:
          loss=t.nn.functional.binary_cross_entropy(y_hat.view(-1),y.to(t.float))
          acc=self.valid_acc(y_hat.view(-1),y)

        # auroc=self.valid_AUROC(y_hat,y)
        # auroc.detach()

        self.log('Validation accuracy',acc)
        # self.log('Validation AUROC',auroc)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = t.optim.Adam(self.parameters(), lr=0.0001,weight_decay=0.0001)
        scheduler= t.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, min_lr=0.00001, patience=5)
        return {'optimizer': optimizer,
                'lr_scheduler': scheduler,
                'monitor': 'train_loss'}

 

def ResNet10_1d(in_channels=14,seq_length=3840,layer_blocks=[1,1,1,1],do_flag=0,num_classes=1):
    return ResNet(in_channels,seq_length,layer_blocks,do_flag,num_classes)

def ResNet18_1d(in_channels=14,seq_length=3840,layer_blocks=[2,2,2,2],do_flag=0,num_classes=1):
    return ResNet(in_channels,seq_length,layer_blocks,do_flag,num_classes)

def ResNet34_1d(in_channels=14,seq_length=3840,layer_blocks=[3,4,6,3],do_flag=0, num_classes=1):
    return ResNet(in_channels,seq_length,layer_blocks,do_flag, num_classes)

#------------InceptionTime------------#

class Inception_block(t.nn.Module):
    def __init__(self, in_ch,kernel_size,bottle_neck_size,filters):
        super(Inception_block,self).__init__()

        self.bottle_neck=t.nn.Conv1d(in_channels=in_ch,out_channels=bottle_neck_size,kernel_size=1,padding=0, stride=1,bias=False)

        kernel_size_list=[kernel_size // (2 ** i)+1 for i in range(3)]

        self.conv_list=t.nn.ModuleList()

        for k_s in kernel_size_list:
            self.conv_list.append(t.nn.Conv1d(in_channels=bottle_neck_size, out_channels=filters,kernel_size=k_s, stride=1,padding=k_s//2,bias=False))
        
        self.maxpool=t.nn.MaxPool1d(kernel_size=3,stride=1,padding=1)

        self.conv_last= t.nn.Conv1d(in_channels=bottle_neck_size,out_channels=filters,kernel_size=1, bias=False)

        self.bn=t.nn.BatchNorm1d(filters*4)

        self.relu=t.nn.ReLU()
        
    def forward(self,x):

        x=self.bottle_neck(x)

        out_list=[]
        for conv in self.conv_list:
            out_list.append(conv(x))
        x=self.maxpool(x)
        out_list.append(self.conv_last(x))

        x=t.cat(out_list,dim=1)
        x=self.bn(x)
        x=self.relu(x)
        return x

class InceptionTime(pl.LightningModule):
    def __init__(self,seq_len,in_ch,kernel_size,bottle_neck_size,filters,num_classes):
        super(InceptionTime,self).__init__()
        self.num_classes=num_classes

        self.incep1=Inception_block(in_ch,kernel_size,bottle_neck_size,filters)
        self.incep2=Inception_block(filters*4,kernel_size,bottle_neck_size,filters)
        self.incep3=Inception_block(filters*4,kernel_size,bottle_neck_size,filters)
        self.incep4=Inception_block(filters*4,kernel_size,bottle_neck_size,filters)
        self.incep5=Inception_block(filters*4,kernel_size,bottle_neck_size,filters)
        self.incep6=Inception_block(filters*4,kernel_size,bottle_neck_size,filters)
        
        self.conv1=t.nn.Conv1d(in_channels=in_ch,out_channels=filters*4,kernel_size=1,padding=0, stride=1,bias=False)

        self.conv2=t.nn.Conv1d(in_channels=filters*4,out_channels=filters*4,kernel_size=1,padding=0, stride=1,bias=False)

        self.bn1=t.nn.BatchNorm1d(filters*4)

        self.bn2=t.nn.BatchNorm1d(filters*4)

        self.relu=t.nn.ReLU()

        self.avgpool=t.nn.AvgPool1d(kernel_size=seq_len)

        self.fc=t.nn.Linear(filters*4,num_classes)

        if num_classes>1: 
           self.out_a=t.nn.Softmax(dim=1)
        else:
           self.out_a=t.nn.Sigmoid()

        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        
    def forward(self,x):
        x1=self.incep1(x)
        x1=self.incep2(x1)
        x1=self.incep3(x1)

        x2=self.conv1(x)
        x2=self.bn1(x2)
        x=x2+x1
        x=self.relu(x)

        x1=self.incep4(x)
        x1=self.incep5(x1)
        x1=self.incep6(x1)

        x2=self.conv2(x)
        x2=self.bn2(x2)
        x=x2+x1
        x=self.relu(x)
        x=self.avgpool(x)
        x=self.fc(x.view(-1,x.shape[1]))
        x=self.out_a(x)
        return x


    def training_step(self, train_batch):
        x,y=train_batch
        y_hat=self.forward(x)

        if self.num_classes>1:
          loss=t.nn.functional.cross_entropy(y_hat,y)
          acc=self.train_acc(y_hat,y)
        else:
          loss=t.nn.functional.binary_cross_entropy(y_hat.view(-1),y.to(t.float))
          acc=self.train_acc(y_hat.view(-1),y)

        # auroc=self.train_AUROC(y_hat,y)
        # auroc.detach()

        self.log('Train accuracy',acc)
        # self.log('Train AUROC',auroc)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch):
        x,y=val_batch
        y_hat=self.forward(x)

        if self.num_classes>1:
          loss=t.nn.functional.cross_entropy(y_hat,y)
          acc=self.valid_acc(y_hat,y)
        else:
          loss=t.nn.functional.binary_cross_entropy(y_hat.view(-1),y.to(t.float))
          acc=self.valid_acc(y_hat.view(-1),y)

        # auroc=self.valid_AUROC(y_hat,y)
        # auroc.detach()

        self.log('Validation accuracy',acc)
        # self.log('Validation AUROC',auroc)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = t.optim.Adam(self.parameters(), lr=0.0001,weight_decay=0.0001)
        scheduler= t.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, min_lr=0.00001, patience=5)
        return {'optimizer': optimizer,
                'lr_scheduler': scheduler,
                'monitor': 'train_loss'}





#-------------EEGNet---------------#
class Inception_eeg(t.nn.Module):
    def __init__(self, in_ch,bottle_neck_size,filters):
        super(Inception_eeg,self).__init__()

        self.bottle_neck=t.nn.Conv1d(in_channels=in_ch,out_channels=bottle_neck_size,kernel_size=1,padding=0, stride=1,bias=False)

        kernel_size_list1=[41, 21, 11]
        kernel_size_list2=[21, 11, 11, 5, 3]
        self.conv_list1=t.nn.ModuleList()
        self.conv_list2=t.nn.ModuleList()

        for k_s in kernel_size_list1:
            self.conv_list1.append(t.nn.Conv1d(in_channels=bottle_neck_size, out_channels=filters,kernel_size=k_s, stride=1,padding=k_s//2,bias=False))

        for k_s in kernel_size_list2:
            self.conv_list2.append(t.nn.Conv1d(in_channels=filters, out_channels=filters,kernel_size=k_s, stride=1,padding=k_s//2,bias=False))

        self.bn=t.nn.BatchNorm1d(filters*6)

        self.relu=t.nn.ReLU()
        
    def forward(self,x):

        x=self.bottle_neck(x)

        out_list=[]
        x41=self.conv_list1[0](x)
        x21_1=self.conv_list1[1](x)
        x11_1=self.conv_list1[2](x)
        
        out_list.append(x41)
        out_list.append(x21_1)
        out_list.append(x11_1)

        x21_2=self.conv_list2[0](x41)
        x11_2=self.conv_list2[1](x41)

        out_list.append(x11_2)

        x11_3=self.conv_list2[2](x21_2)
        x5=self.conv_list2[3](x21_2)
        x3=self.conv_list2[4](x11_2)

        out_list.append(x5)
        out_list.append(x3)

        x=t.cat(out_list,dim=1)
        x=self.bn(x)
        x=self.relu(x)
        return x

class EEG_net(pl.LightningModule):
    def __init__(self,seq_len,in_ch,bottle_neck_size,filters,num_classes):
        super(EEG_net,self).__init__()
        self.num_classes=num_classes

        self.incep1=Inception_eeg(in_ch,bottle_neck_size,filters)
        self.incep2=Inception_eeg(filters*6,bottle_neck_size,filters)
        self.incep3=Inception_eeg(filters*6,bottle_neck_size,filters)
        self.incep4=Inception_eeg(filters*6,bottle_neck_size,filters)
        self.incep5=Inception_eeg(filters*6,bottle_neck_size,filters)
        self.incep6=Inception_eeg(filters*6,bottle_neck_size,filters)
        
        self.conv1=t.nn.Conv1d(in_channels=in_ch,out_channels=filters*6,kernel_size=1,padding=0, stride=1,bias=False)

        self.conv2=t.nn.Conv1d(in_channels=filters*6,out_channels=filters*6,kernel_size=1,padding=0, stride=1,bias=False)

        self.bn1=t.nn.BatchNorm1d(filters*6)

        self.bn2=t.nn.BatchNorm1d(filters*6)

        self.relu=t.nn.ReLU()

        self.avgpool=t.nn.AvgPool1d(kernel_size=seq_len)

        self.fc=t.nn.Linear(filters*6,num_classes)

        if num_classes>1: 
           self.out_a=t.nn.Softmax(dim=1)
        else:
           self.out_a=t.nn.Sigmoid()

        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        
    def forward(self,x):
        x1=self.incep1(x)
        x1=self.incep2(x1)
        x1=self.incep3(x1)

        x2=self.conv1(x)
        x2=self.bn1(x2)
        x=x2+x1
        x=self.relu(x)

        x1=self.incep4(x)
        x1=self.incep5(x1)
        x1=self.incep6(x1)

        x2=self.conv2(x)
        x2=self.bn2(x2)
        x=x2+x1
        x=self.relu(x)
        x=self.avgpool(x)
        x=self.fc(x.view(-1,x.shape[1]))
        x=self.out_a(x)
        return x
  
    def training_step(self, train_batch):
        x,y=train_batch
        y_hat=self.forward(x)

        if self.num_classes>1:
          loss=t.nn.functional.cross_entropy(y_hat,y)
          acc=self.train_acc(y_hat,y)
        else:
          loss=t.nn.functional.binary_cross_entropy(y_hat.view(-1),y.to(t.float))
          acc=self.train_acc(y_hat.view(-1),y)

        # auroc=self.train_AUROC(y_hat,y)
        # auroc.detach()

        self.log('Train accuracy',acc)
        # self.log('Train AUROC',auroc)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch):
        x,y=val_batch
        y_hat=self.forward(x)

        if self.num_classes>1:
          loss=t.nn.functional.cross_entropy(y_hat,y)
          acc=self.valid_acc(y_hat,y)
        else:
          loss=t.nn.functional.binary_cross_entropy(y_hat.view(-1),y.to(t.float))
          acc=self.valid_acc(y_hat.view(-1),y)

        # auroc=self.valid_AUROC(y_hat,y)
        # auroc.detach()

        self.log('Validation accuracy',acc)
        # self.log('Validation AUROC',auroc)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = t.optim.Adam(self.parameters(), lr=0.0001,weight_decay=0.0001)
        scheduler= t.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, min_lr=0.00001, patience=5)
        return {'optimizer': optimizer,
                'lr_scheduler': scheduler,
                'monitor': 'train_loss'}