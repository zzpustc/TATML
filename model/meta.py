import  paddle
import  paddle.nn as nn
import paddle.optimizer as optim
from    paddle.nn import functional as F
import paddle.fluid as fluid
import  numpy as np

from    model.learner_new import Learner
from    copy import deepcopy
import pickle



class Meta(fluid.dygraph.Layer):
    """
    Meta Learner
    """
    def __init__(self, args, config):
        """

        :param args:
        """
        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_qry = args.k_qry
        self.task_num = args.train_tasks
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test

        self.criterion = nn.KLDivLoss()
        #self.net = Learner(config, args.imgc, args.imgw, args.imgh)
        self.net = Learner()
        self.schedule = optim.lr.MultiStepDecay(learning_rate=self.meta_lr, milestones = [15, 30], gamma=0.1)
        self.meta_optim = optim.Adam(parameters = self.net.parameters(), learning_rate=self.schedule, weight_decay = 5.0e-4)




    def EntropyLoss(self, predict_prob, class_level_weight=None, instance_level_weight=None, epsilon= 1e-20):

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

        entropy = -predict_prob*paddle.log(predict_prob + epsilon) - (1 - predict_prob)*paddle.log(1- predict_prob + epsilon)

        return paddle.sum(instance_level_weight * entropy * class_level_weight) / float(N)
    
    def semi_forward(self, x_spt, y_spt, x_qry, y_qry_hard, y_qry_soft, bal_ratio_qry, epoch):
        """

        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        x_spt, y_spt = x_spt.unsqueeze(0), y_spt.unsqueeze(0)
        self.schedule.step(epoch=epoch)
        task_num, setsz, c_, h, w = x_spt.shape
        querysz = (x_qry.shape)[1]

        bal_ratio_qry = bal_ratio_qry.reshape([task_num, -1])

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i

        for i in range(task_num):
            
            # 1. run the i-th task and compute loss for k=0
            logits = self.net(x_spt[i], params=None, bn_training=True)

            loss_cls = F.cross_entropy(logits, y_spt[i].cast('int64'))
            
            loss = loss_cls

            grad0 = paddle.grad(loss, self.net.parameters())

            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad0, self.net.parameters())))

            # this is the loss and accuracy before first update
            with paddle.no_grad():
                # [setsz, nway]
                logits = self.net(x_qry[i].cast('float32'), self.net.parameters(), bn_training=True)

                logits_kl = F.log_softmax(logits, axis=1)
            
                loss_kl = self.criterion(logits_kl, y_qry_soft[i])
                loss_cls = F.cross_entropy(logits, y_qry_hard[i].squeeze().cast('int64'), reduction='none')
                loss_cls = (loss_cls * bal_ratio_qry[i]).mean()
            
                loss_pseudo = loss_cls + loss_kl
            
                logits_c = F.softmax(logits, axis=1)
                loss_enpy = self.EntropyLoss(logits_c)
                loss_q = loss_pseudo + loss_enpy
            
                losses_q[0] += loss_q
             

            # this is the loss and accuracy after the first update

            logits = self.net(x_qry[i].cast('float32'), fast_weights, bn_training=True)
                
            logits_kl = F.log_softmax(logits, axis=1)
            loss_kl = self.criterion(logits_kl, y_qry_soft[i])
            loss_cls = F.cross_entropy(logits, y_qry_hard[i].squeeze().cast('int64'), reduction='none').mean()
            loss_cls = (loss_cls * bal_ratio_qry[i]).mean()
            loss_pseudo = loss_cls + loss_kl
            
            logits_c = F.softmax(logits, axis=1)
            loss_enpy = self.EntropyLoss(logits_c)
            
            loss_q = loss_pseudo + loss_enpy          

            losses_q[1] += loss_q

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.net(x_spt[i].cast('float32'), fast_weights, bn_training=True)
          
                loss_cls = F.cross_entropy(logits, y_spt[i].cast('int64'))
            
                loss = loss_cls
            
                # 2. compute grad on theta_pi
                grad = paddle.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                logits = self.net(x_qry[i].cast('float32'), fast_weights, bn_training=True)
                
                logits_kl = F.log_softmax(logits, axis=1)
                loss_kl = self.criterion(logits_kl, y_qry_soft[i])
                loss_cls = F.cross_entropy(logits, y_qry_hard[i].squeeze().cast('int64'), reduction='none').mean()
                loss_cls = (loss_cls * bal_ratio_qry[i]).mean()
              #  loss_cls = (loss_cls).mean()
                loss_pseudo = loss_cls + loss_kl
            
                logits_c = F.softmax(logits, axis=1)
                loss_enpy = self.EntropyLoss(logits_c)
            
                loss_q = loss_pseudo + loss_enpy                    
                
                # loss_q will be overwritten and just keep the loss_q on last update step.
                losses_q[k + 1] += loss_q 
    
        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num

        # optimize theta parameters
        self.meta_optim.clear_grad()
        loss_q.backward()
 #       nn.utils.clip_grad_norm(self.net.parameters(), max_norm=100, norm_type=2)
        self.meta_optim.step()

        return   


    def forward(self, x_spt, y_spt, x_qry, y_qry, epoch):
        """

        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """  
        
        self.schedule.step(epoch=epoch)
        task_num, setsz, c_, h, w = x_spt.shape
        querysz = (x_qry.shape)[1]

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]


        for i in range(task_num):

            # 1. run the i-th task and compute loss for k=0
            logits = self.net(x_spt[i], params=None, bn_training=True)
            loss = F.cross_entropy(logits, y_spt[i].cast('int64'))
            grad = paddle.grad(loss, self.net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

            # this is the loss and accuracy before first update
            with paddle.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i].cast('int64'))
                losses_q[0] += loss_q

                pred_q = F.softmax(logits_q, axis=1).argmax(axis=1)
                correct = paddle.equal(pred_q, y_qry[i].cast('int64')).numpy().sum().item()
                corrects[0] = corrects[0] + correct

            # this is the loss and accuracy after the first update
            with paddle.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i].cast('int64'))
                losses_q[1] += loss_q
                # [setsz]
                pred_q = F.softmax(logits_q, axis=1).argmax(axis=1)
                correct = paddle.equal(pred_q, y_qry[i].cast('int64')).numpy().sum().item()
                corrects[1] = corrects[1] + correct

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.net(x_spt[i], fast_weights, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i].cast('int64'))
                # 2. compute grad on theta_pi
                grad = paddle.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = F.cross_entropy(logits_q, y_qry[i].cast('int64'))
                losses_q[k + 1] += loss_q

                with paddle.no_grad():
                    pred_q = F.softmax(logits_q, axis=1).argmax(axis=1)
                    correct = paddle.equal(pred_q, y_qry[i].cast('int64')).numpy().sum().item()
                    corrects[k + 1] = corrects[k + 1] + correct



        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num

        # optimize theta parameters
        self.meta_optim.clear_grad()
        loss_q.backward()
        self.meta_optim.step()
        accs = np.array(corrects) / (querysz * task_num)

        return accs


    def finetunning(self, args, x_spt, y_spt, x_qry, y_qry, best_acc):
        """

        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        assert len(x_spt.shape) == 4

        querysz = (x_qry.shape)[0]

        accs = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt)
        loss = F.cross_entropy(logits, y_spt.cast('int64'))
        grad = paddle.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        acc = self.test(net, fast_weights, x_qry, y_qry)
        accs[1] = acc

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits  = net(x_spt, fast_weights, bn_training=True)
            loss = F.cross_entropy(logits, y_spt.cast('int64'))
            # 2. compute grad on theta_pi
            grad = paddle.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            acc = self.test(net, fast_weights, x_qry, y_qry)
            accs[k+1] = acc

        if accs[-1] > best_acc and args.save:
            best_acc = accs[-1]
            model_params = [item.numpy() for item in net.vars]
            model_params_file = open('./weights/maml.pkl', 'wb')
            pickle.dump(model_params, model_params_file)
            model_params_file.close()

        del net

        return accs
        
    def test(self, net, fast_weights, x_qry, y_qry):
        
        l = (x_qry.shape)[0]
        em_len = l // 4
        
        corrects = 0
        
        for i in range(em_len):
            x_qry_tmp = x_qry[4*i:4*(i+1)]
            y_qry_tmp = y_qry[4*i:4*(i+1)].squeeze()
            
            logits_q = net(x_qry_tmp, params=fast_weights, bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, axis=1).argmax(axis=1)
            correct = paddle.equal(pred_q, y_qry[i].cast('int64')).numpy().sum().item()
            
            corrects += correct
            
        acc = corrects / l
        
        return acc
            


def main():
    pass


if __name__ == '__main__':
    main()
