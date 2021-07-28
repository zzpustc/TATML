import paddle, os
import paddle.nn as nn
import paddle.fluid as fluid
from    paddle.nn import functional as F
import paddle.optimizer as optim
import  numpy as np
import  random, sys, pickle
import  argparse

from datasets.HAR_new import HAR
#from datasets.HAR_task import HAR
#from datasets.HAR_person import HAR
from model.meta import Meta

def main(args):
    # fix the seed and promise reproduction
    seed = 222
    paddle.seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    print(args)

    config = [
        ('conv2d', [32, 3, 11, 11, 1, 0]),
        ('max_pool2d', [2, 2, 0]),
        ('bn', [32]),
        ('relu', [True]),
        ('conv2d', [32, 32, 7, 7, 1, 0]),
        ('max_pool2d', [2, 2, 0]),
        ('bn', [32]),
        ('relu', [True]),
        ('conv2d', [32, 32, 7, 7, 1, 0]),
        ('bn', [32]),
        ('relu', [True]),
        ('max_pool2d', [2, 2, 0]),
        ('flatten', []),
        ('linear', [32 * 26* 19, args.n_way]),
    ]

    use_gpu = True
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    
    # define meta model (student network)   
    maml = Meta(args, config)

    maml.net.train()
    
    best_acc = 0
    best_acc_wo = 0
    for epoch in range(args.epoch):
        print('Epoch:', epoch)
        
        # define dataset
        har = HAR(args)
        train_loader = paddle.io.DataLoader(
        har,
        batch_size = args.train_tasks,
        shuffle = True,
        drop_last=False,
        num_workers=args.num_workers
        )
        ts_spt_x, ts_spt_y = har.get_finetune_item()
        ts_spt_x, ts_spt_y = paddle.to_tensor(ts_spt_x), paddle.to_tensor(ts_spt_y).squeeze()
        ts_spt_x.stop_gradient = False
        ts_spt_y.stop_gradient = False
        
        print('*****************************Training**********************************')

        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(train_loader):

            x_spt, y_spt, x_qry, y_qry = paddle.to_tensor(x_spt), paddle.to_tensor(y_spt), paddle.to_tensor(x_qry), paddle.to_tensor(y_qry)
            x_spt.stop_gradient = False
            y_spt.stop_gradient = False
            x_qry.stop_gradient = False
            y_qry.stop_gradient = False
            accs = maml(x_spt, y_spt, x_qry, y_qry, epoch)
            if step % 10 == 0:
                print('step:', step, '\ttraining acc:', accs)
            
            if step % 100 == 0:  # evaluation
                ts_x, ts_y = har.get_trg_item()
                ts_x, ts_y = paddle.to_tensor(ts_x), paddle.to_tensor(ts_y)
                accs = maml.finetunning(args, ts_spt_x, ts_spt_y, ts_x, ts_y, best_acc)

                if accs[0] > best_acc_wo:
                    best_acc_wo = accs[0]

                if accs[-1] > best_acc:
                    best_acc = accs[-1]

                print('MAML:', args.src_id, 'to', args.trg_id)
                print('Test acc:', accs)
                print('Best acc w/o ft:', best_acc_wo)
                print('Best acc:', best_acc)

      #          f = open('./tmp/results.txt', 'a')
      #          domain_infos = 'MAML ' + str(args.src_id) +' '+ 'to' + ' ' + str(args.trg_id) + ' '
      #          f.write(domain_infos) 
      #          f.write(str(best_acc_wo)) 
      #          f.write('   ')
      #          f.write(str(best_acc)) 
      #          f.write('\n')
      #          f.close()    
        

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=10)
    argparser.add_argument('--batchsz', type=int, help='number of the sampling tasks', default=1000)
    argparser.add_argument('--train_tasks', type=int, help='train_tasks one time', default=2)
    argparser.add_argument('--num_workers', type=int, help='Number of workers when reading data', default=0)
    argparser.add_argument('--print_log', type=int, help='The interval of printing', default=125)
    argparser.add_argument('--n_way', type=int, help='n way', default=4)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--imgw', type=int, help='imgw', default=260)
    argparser.add_argument('--imgh', type=int, help='imgw', default=200)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.001)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=4)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=5)
    argparser.add_argument('--src_id', type=str, help='Source domain ID, interger or str list', default='2')
    argparser.add_argument('--trg_id', type=str, help='Target domain ID', default='1')
    argparser.add_argument('--root_path', type=str, help='The root path of HAR dataset', default='/home/rookiezhi/zzp/data/HAR/Processed/')
    argparser.add_argument('--transform', type=bool, help='If transform the data', default=False)
    argparser.add_argument('--save', type=bool, help='Save the model weights', default=True)
    argparser.add_argument('--multi-source', type=bool, help='if multi-source DA', default=True)
    argparser.add_argument('--task_type', type=str, help='task type selection', default='else')
    argparser.add_argument('--per_id', type=str, help='Person ID', default='1')

    
    args = argparser.parse_args()

    main(args)
