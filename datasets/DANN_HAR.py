import paddle
import numpy as np
import os  
import paddle.nn.functional as F
from paddle.vision import transforms
from paddle.io import Dataset
import paddle.fluid as fluid
import random

from numpy.random import choice

#from datasets.data_augmentation import GaussianBlur,PacketMissing,ComponentExchange,ComponentErase 

class HAR_tar(Dataset):
    def __init__(self, args):

       self.transform = args.transform
       self.n_way = args.n_way
       self.k_spt = args.k_spt
       self.k_qry = args.k_qry
       self.imgw = args.imgw
       self.imgh = args.imgh
       self.imgc = args.imgc
       self.setsz = self.n_way * self.k_spt  # num of samples per set
       self.querysz = self.n_way * self.k_qry  # number of samples per set for evaluation

       if args.multi_source:
           src_csi_set = []
           src_env_set = []
           src_per_set = []
           src_act_set = []
           for sid in list(args.src_id):   
               src_csi_tmp = np.load(args.root_path + sid + '/' + 'csi_seg.npy')
               src_env_tmp = np.load(args.root_path + sid + '/' + 'envir_label.npy')
               src_per_tmp = np.load(args.root_path + sid + '/' + 'per_label.npy')
               src_act_tmp = np.load(args.root_path + sid + '/' + 'act_label.npy')  
               src_csi_set.append(src_csi_tmp)
               src_env_set.append(src_env_tmp)
               src_per_set.append(src_per_tmp)
               src_act_set.append(src_act_tmp)

           self.src_csi = np.concatenate(src_csi_set, 0)
           self.src_env = np.concatenate(src_env_set, 0)
           self.src_per = np.concatenate(src_per_set, 0)
           self.src_act = np.concatenate(src_act_set, 0)


       else:
               self.src_csi = np.load(args.root_path + str(args.src_id) + '/' + 'csi_seg.npy')
               self.src_env = np.load(args.root_path + str(args.src_id) + '/' + 'envir_label.npy')
               self.src_per = np.load(args.root_path + str(args.src_id) + '/' + 'per_label.npy')
               self.src_act = np.load(args.root_path + str(args.src_id) + '/' + 'act_label.npy')
       
       trg_csi_set = []
       trg_env_set = []
       trg_per_set = []
       trg_act_set = []
       for tid in list(args.trg_id):   
               trg_csi_tmp = np.load(args.root_path + tid + '/' + 'csi_seg.npy')
               trg_env_tmp = np.load(args.root_path + tid + '/' + 'envir_label.npy')
               trg_per_tmp = np.load(args.root_path + tid + '/' + 'per_label.npy')
               trg_act_tmp = np.load(args.root_path + tid + '/' + 'act_label.npy')  
               trg_csi_set.append(trg_csi_tmp)
               trg_env_set.append(trg_env_tmp)
               trg_per_set.append(trg_per_tmp)
               trg_act_set.append(trg_act_tmp)

       self.trg_csi = np.concatenate(trg_csi_set, 0)
       self.trg_env = np.concatenate(trg_env_set, 0)
       self.trg_per = np.concatenate(trg_per_set, 0)
       self.trg_act = np.concatenate(trg_act_set, 0) 

       self.train_transform = transforms.Compose(
            [
                # GaussianBlur(),
                # PacketMissing(),  
                # ComponentExchange(),
                # ComponentErase(),
            ]
        ) 
       
       self.ft_csi, self.ft_act, self.ts_csi, self.ts_act = self.select_sup()
            
        
    # select n_shot from each class of the target domain
    def select_sup(self):
        """
        Get the support set and the test samples from the target domain
        """
        sup_idx = []
        qry_idx = []
        for a in range(1,self.n_way+1):
            act_idx = np.nonzero((self.trg_act==a))[0]
            sup_idx.extend(act_idx[0:self.k_spt])
            qry_idx.extend(act_idx[self.k_spt:])
                
        finetune_csi = self.trg_csi[sup_idx]    
        finetune_act = self.trg_act[sup_idx] - 1
        
        test_csi = self.trg_csi[qry_idx]
        test_act = self.trg_act[qry_idx] - 1
        
        
        return finetune_csi, finetune_act, test_csi, test_act

    def __getitem__(self, idx):

        spt_x_img = self.trg_csi[idx]
        o = self.resize_py(spt_x_img)
        support_x = o

        spt_y = self.trg_act[idx] - 1
        support_y = spt_y  
       
        return support_x, support_y

    def get_finetune_item(self):
        
       finetune_csi = self.ft_csi
       new_csi = np.ones(shape=finetune_csi.shape)
       new_csi = np.tile(new_csi.view(finetune_csi.shape[0],1,260,200),[1,3,1,1])
       for i in range(finetune_csi.shape[0]):
           t = finetune_csi[i]
           o = self.resize_py(t)#.repeat(3,1,1)
          # o = self.train_transform(t.float()).repeat(3,1,1)
           new_csi[i] = o
       finetune_act = self.ft_act
       
       return new_csi, finetune_act
       
    def get_trg_item(self):
       
       trg_csi = self.ts_csi
       new_csi = np.ones(shape=trg_csi.shape)
       new_csi = np.tile(new_csi.view(trg_csi.shape[0],1,260,200),[1,3,1,1])
       for i in range(trg_csi.shape[0]):
           t = trg_csi[i]
           o = self.resize_py(t)#.repeat(3,1,1)
          # o = self.train_transform(t.float()).repeat(3,1,1)
           new_csi[i] = o
       trg_act = self.ts_act
       
       return new_csi, trg_act       
   
    def resize_py(self, t):
        tmp = np.tile(t.reshape((40, 1300)).reshape((40,20,65)).transpose([1,2,0]).reshape((20,1,65,40)),[1,3,1,1])
        o1  = np.concatenate([tmp[i] for i in range(0,5)],axis=2)
        o2  = np.concatenate([tmp[i] for i in range(5,10)],axis=2)
        o3  = np.concatenate([tmp[i] for i in range(10,15)],axis=2)
        o4  = np.concatenate([tmp[i] for i in range(15,20)],axis=2)
        
        o = np.concatenate([o1,o2,o3,o4], axis=1)
        return o

    def __len__(self):
        return len(self.trg_csi)
