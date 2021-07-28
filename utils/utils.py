import paddle, os
import paddle.nn as nn
from    paddle.nn import functional as F
import  numpy as np
import  random, sys, pickle
import  argparse
from copy import deepcopy

from model.meta import Meta

# Compute information entropy
def EntropyLoss(predict_prob, class_level_weight=None, instance_level_weight=None, epsilon= 1e-20):

        N, C = predict_prob.shape
    
        if class_level_weight is None:
            class_level_weight = 1.0
        else:
            if len(class_level_weight.size()) == 1:
                class_level_weight = class_level_weight.view(1, class_level_weight.size(0))
            assert class_level_weight.size(1) == C, 'fatal error: dimension mismatch!'
        
        if instance_level_weight is None:
            instance_level_weight = 1.0
        else:
            if len(instance_level_weight.size()) == 1:
                instance_level_weight = instance_level_weight.view(instance_level_weight.size(0), 1)
            assert instance_level_weight.size(0) == N, 'fatal error: dimension mismatch!'

        entropy = -predict_prob*paddle.log(predict_prob + epsilon) - (1 - predict_prob)*paddle.log(1 - predict_prob + epsilon)
    #    entropy = -torch.log(predict_prob + epsilon)- torch.log(1 - predict_prob + epsilon)
        return paddle.sum(entropy, 1)

# get pseudo label with trained maml    
def Pseudo_label(args, model, x_qry):
    y_qry_hard = paddle.ones([1, args.k_qry * args.n_way, 1])
    y_qry_soft = paddle.ones([1, args.k_qry * args.n_way, args.n_way])
    bal_ratio_qry = paddle.ones([1, args.k_qry * args.n_way])
    
    # copy and resume the model
    model_cp = deepcopy(model)
    model_params_file = open(args.resume ,'rb')
    model_params = pickle.load(model_params_file)
    model_params_file.close()
    model_cp.vars = [paddle.to_tensor(item, stop_gradient=False) for item in model_params]

    for t in range(1):        
        y_qry = model_cp(x_qry[t].cast('float32'))
        y_qry = F.softmax(y_qry,1)
        enpy_qry = EntropyLoss(y_qry)

        bal_ratio_qry[t] = args.alpha * paddle.exp(-enpy_qry)
        
        y_qry_soft[t] = y_qry
        y_qry_hard[t,:,0] = y_qry.max(1)[1]
        
    del model_cp
        
    return y_qry_hard.cast('int32'), y_qry_soft, bal_ratio_qry

def resign(args, x_spt, y_spt, x_qry, y_qry, model):
    '''   
    Select the most representative samples to enter into the support set
    '''
    
    _,_,c,h,w = x_spt.shape
    x_spt_oneshot = x_spt
    y_spt_oneshot = y_spt
    x_qry_oneshot = x_qry
    y_qry_oneshot = y_qry

    x_spt = x_spt.reshape([args.train_tasks, args.n_way, args.k_spt, c, h, w]) # 10*5*1*3*84*84
    x_qry = x_qry.reshape([args.train_tasks, args.n_way, args.k_qry, c, h, w]) # 10*5*15*3*84*84
    
    y_spt = y_spt.reshape([args.train_tasks, args.n_way, args.k_spt]) 
    y_qry = y_qry.reshape([args.train_tasks, args.n_way, args.k_qry])
        
    x_batch = paddle.concat((x_spt, x_qry), 2)
    y_batch = paddle.concat((y_spt, y_qry), 2)
    
    score = paddle.zeros_like(x_spt).cuda()
    
    x_selective_spt = paddle.zeros_like(x_spt).cuda()
    x_selective_qry = paddle.zeros_like(x_qry).cuda()
    y_selective_spt = paddle.zeros_like(y_spt).cuda()
    y_selective_qry = paddle.zeros_like(y_qry).cuda()
    
    template = paddle.zeros([args.n_way, args.k_spt + args.k_qry]).cuda()
    
    for i in range(args.train_tasks):
        x_tmp = x_batch[i]
        y_tmp = y_batch[i]
        x_tmp = x_tmp.reshape([-1, c, h, w])
        out  = model.net(x_tmp, params=None, bn_training=True)
        loss = F.cross_entropy(out,y_tmp.reshape([-1]).cast('int64'))
        out = F.softmax(out, 1)
        entropy_pre = EntropyLoss(out).reshape([args.n_way,-1])

        grad = paddle.grad(loss, model.net.parameters())
        fast_weights = list(map(lambda p: p[1] - 0.01*p[0], zip(grad, model.net.parameters())))

        out  = model.net(x_tmp, fast_weights, bn_training=True)

        out = F.softmax(out, 1)
        entropy = EntropyLoss(out).reshape([args.n_way,-1])

        entropy_diff = entropy_pre - entropy
        idx = paddle.argsort(entropy_pre, 1, descending=True)
        
        order_list = [i for i in range(args.k_spt+args.k_qry)]
        
        
        if args.k_spt > 1:     
            cur = int((args.k_spt+args.k_qry) / args.k_spt)
            spt_list = [x*cur for x in range(args.k_spt)]
            qry_list = [x for x in order_list if x not in spt_list]
            
            spt_idx = idx[:, spt_list]
            qry_idx = idx[:, qry_list]    
            
        else:
            spt_list = [int((args.k_spt+args.k_qry)/2)]
            qry_list = [x for x in order_list if x not in spt_list]
            
            idx_np = idx.numpy()
            
            spt_idx = idx_np[:, spt_list]
            qry_idx = idx_np[:, qry_list]
        
        
        x_tmp = x_tmp.reshape([args.n_way, -1, c, h, w])
        for j in range(args.n_way):
            new_x_spt = x_tmp[j,int(spt_idx[j,0]), :,:,:]
            new_y_spt = y_tmp[j,int(spt_idx[j,0])]
            
            if int(spt_idx[j,0]) == 0:
                x_qry_seq2 = x_tmp[j,int(spt_idx[j,0])+1:, :,:,:]
                y_qry_seq2 = y_tmp[j,int(spt_idx[j,0])+1:]
                new_x_qry = x_qry_seq2
                new_y_qry = y_qry_seq2
            elif int(spt_idx[j,0]) == (args.k_spt + args.k_qry - 1):
                x_qry_seq1 = x_tmp[j,0:int(spt_idx[j,0]), :,:,:]
                y_qry_seq1 = y_tmp[j,0:int(spt_idx[j,0])]
                new_x_qry = x_qry_seq1
                new_y_qry = y_qry_seq1
            else:
                x_qry_seq1 = x_tmp[j,0:int(spt_idx[j,0]), :,:,:]
                y_qry_seq1 = y_tmp[j,0:int(spt_idx[j,0])]
                x_qry_seq2 = x_tmp[j,int(spt_idx[j,0])+1:, :,:,:]
                y_qry_seq2 = y_tmp[j,int(spt_idx[j,0])+1:]
                new_x_qry = paddle.concat([x_qry_seq1, x_qry_seq2], axis=0)
                new_y_qry = paddle.concat([y_qry_seq1, y_qry_seq2], axis=0)
            
            x_selective_spt[i,j] = new_x_spt
            y_selective_spt[i,j] = new_y_spt
            x_selective_qry[i,j] = new_x_qry
            y_selective_qry[i,j] = new_y_qry

    x_selective_spt = x_selective_spt.reshape([args.train_tasks, -1, c, h, w])
    x_selective_qry = x_selective_qry.reshape([args.train_tasks, -1, c, h, w])
    y_selective_spt = y_selective_spt.reshape([args.train_tasks, -1])
    y_selective_qry = y_selective_qry.reshape([args.train_tasks, -1])
     
    
    return x_selective_spt, y_selective_spt, x_selective_qry, y_selective_qry

