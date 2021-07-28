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

class HAR(Dataset):
    def __init__(self, args):

       self.transform = args.transform
       self.n_way = args.n_way
       self.batchsz = args.batchsz
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
                GaussianBlur(),
                PacketMissing(),  
                ComponentExchange(),
                ComponentErase(),
            ]
        )   
       
       self.ft_csi, self.ft_act, self.ts_csi, self.ts_act = self.select_sup()
       
       self.create_batch(self.batchsz)
       
    def create_batch(self, batchsz):
        """
        create batch for meta-learning.
        """
        self.support_x_batch = []  # support set batch
        self.query_x_batch = []  # query set batch
        self.support_y_batch = []  # support set batch
        self.query_y_batch = []  # query set batch
        for b in range(batchsz):  # for each batch
            support_x = []
            support_y = []
            query_x = []
            query_y = []
            for cls in range(1, self.n_way+1):
                # 2. select k_shot + k_query for each class
                cls_idx = np.nonzero((self.src_act==cls))[0]
                selected_imgs_idx = np.random.choice(len(cls_idx), self.k_spt + self.k_qry, False)
                np.random.shuffle(selected_imgs_idx)

                indexDtrain = cls_idx[selected_imgs_idx[:self.k_spt]]
                indexDtest = cls_idx[selected_imgs_idx[self.k_spt:]]

                support_x.extend(indexDtrain.tolist())  # get all images filename for current Dtrain
                query_x.extend(indexDtest.tolist())

                # support_x.append(
                #     np.array(self.src_csi)[indexDtrain].tolist())  # get all images filename for current Dtrain
                # support_y.append(
                #     np.array(self.src_act)[indexDtrain].tolist())
                # query_x.append(self.src_csi[indexDtest].tolist())
                # query_y.append(self.src_act[indexDtest].tolist())

            #shuffle the correponding relation between support set and query set
            random.shuffle(support_x)
            random.shuffle(query_x)

            self.support_x_batch.append(support_x)  # append set to current sets
            self.query_x_batch.append(query_x)  # append sets to current sets        
        
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
      
       support_x = paddle.zeros([self.setsz, 3, self.imgw, self.imgh], dtype = "float32")
       query_x = paddle.zeros([self.querysz, 3, self.imgw, self.imgh], dtype = "float32")

       support_y = paddle.zeros((self.setsz, 1), dtype = "float32")
       query_y = paddle.zeros((self.querysz, 1), dtype = "float32")
      
       flatten_support_x = self.support_x_batch[idx]
       flatten_query_x = self.query_x_batch[idx]
       
       if self.transform:
           for i, spt_x in enumerate(flatten_support_x):
                spt_x_img = self.src_csi[spt_x]
                spt_x_img = paddle.to_tensor(spt_x_img).squeeze()
                support_x[i] = self.train_transform(spt_x_img.cast('float32')).tile([1,3,1,1])

                spt_y = self.src_act[spt_x] - 1
                support_y[i,0] = paddle.to_tensor(spt_y).cast('float32')

           for i, qry_x in enumerate(flatten_query_x):
                qry_x_img = self.src_csi[qry_x]
                qry_x_img = paddle.to_tensor(qry_x_img).squeeze()
                query_x[i] = self.train_transform(qry_x_img.cast('float32')).tile([1,3,1,1])

                qry_y = self.src_act[qry_x] - 1
                query_y[i,0] = paddle.to_tensor(qry_y).cast('float32')
       else:
           for i, spt_x in enumerate(flatten_support_x):
                spt_x_img = paddle.to_tensor(self.src_csi[spt_x])
                o = self.resize_py(spt_x_img.cast('float32'))
                support_x[i] = o

                spt_y = self.src_act[spt_x] - 1
                support_y[i,0] = paddle.to_tensor(spt_y).cast('float32')

           for i, qry_x in enumerate(flatten_query_x):
                qry_x_img = paddle.to_tensor(self.src_csi[qry_x])
                o = self.resize_py(qry_x_img.cast('float32'))
                query_x[i] = o

                qry_y = self.src_act[qry_x] - 1
                query_y[i,0] = paddle.to_tensor(qry_y).squeeze().cast('float32')
           
       
       return support_x.cast('float32'), support_y.squeeze().cast('int32'), query_x.cast('float32'), query_y.squeeze().cast('int32')

    def get_finetune_item(self):
        
       finetune_csi = paddle.to_tensor(self.ft_csi)
       new_csi = paddle.ones(shape=finetune_csi.shape, dtype = "float32")
       new_csi = new_csi.reshape((finetune_csi.shape[0],1,260,200)).tile([1,3,1,1])
       for i in range(finetune_csi.shape[0]):
           t = finetune_csi[i]
           o = self.resize_py(t).cast('float32')
          # o = self.train_transform(t.float()).repeat(3,1,1)
           new_csi[i] = o
       finetune_act = paddle.to_tensor(self.ft_act)
       
       return new_csi, finetune_act.cast('int32')
       
    def get_trg_item(self):
       
       trg_csi = paddle.to_tensor(self.ts_csi)
       new_csi = paddle.ones(shape=trg_csi.shape, dtype = "float32")
       new_csi = new_csi.reshape((trg_csi.shape[0],1,260,200)).tile([1,3,1,1])
       for i in range(trg_csi.shape[0]):
           t = trg_csi[i]
           o = self.resize_py(t).cast('float32')
           new_csi[i] = o
       trg_act = paddle.to_tensor(self.ts_act)
       
       return new_csi, trg_act.cast('int32')      
   
    def resize_py(self, t):
        tmp = t.reshape((40, 1300)).reshape((40,20,65)).transpose([1,2,0]).reshape((20,1,65,40)).tile([1,3,1,1])
        o1  = paddle.concat([tmp[i] for i in range(0,5)],axis=2)
        o2  = paddle.concat([tmp[i] for i in range(5,10)],axis=2)
        o3  = paddle.concat([tmp[i] for i in range(10,15)],axis=2)
        o4  = paddle.concat([tmp[i] for i in range(15,20)],axis=2)
        
        o = paddle.concat([o1,o2,o3,o4], axis=1)
        
        return o

    def __len__(self):
        return self.batchsz
