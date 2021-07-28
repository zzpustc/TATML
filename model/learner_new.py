import  paddle
from    paddle import nn
from    paddle.nn import functional as F
import paddle.fluid as fluid
import  numpy as np


class Learner(paddle.nn.Layer):
    def __init__(self):
        super(Learner, self).__init__()
        # All parameters need to be optimized
        self.vars = []
        self.vars_bn = []

        # ------------------------The first conv2d-------------------------
        weight = paddle.static.create_parameter(shape=[32, 3, 11, 11],
                                                dtype='float32',
                                                default_initializer=nn.initializer.KaimingNormal(),  
                                                is_bias=False)
        bias = paddle.static.create_parameter(shape=[32],
                                              dtype='float32',
                                              is_bias=True) 
        self.vars.extend([weight, bias])
        # The first BatchNorm
        weight = paddle.static.create_parameter(shape=[32],
                                                dtype='float32',
                                                default_initializer=nn.initializer.Constant(value=1), 
                                                is_bias=False)
        bias = paddle.static.create_parameter(shape=[32],
                                              dtype='float32',
                                              is_bias=True)  
        self.vars.extend([weight, bias])
        running_mean = paddle.to_tensor(np.zeros([32], np.float32), stop_gradient=True)
        running_var = paddle.to_tensor(np.zeros([32], np.float32), stop_gradient=True)
        self.vars_bn.extend([running_mean, running_var])

        # ------------------------The second conv2d------------------------
        weight = paddle.static.create_parameter(shape=[32, 32, 7, 7],
                                                dtype='float32',
                                                default_initializer=nn.initializer.KaimingNormal(),
                                                is_bias=False)
        bias = paddle.static.create_parameter(shape=[32],
                                              dtype='float32',
                                              is_bias=True)
        self.vars.extend([weight, bias])
        # The second BatchNorm
        weight = paddle.static.create_parameter(shape=[32],
                                                dtype='float32',
                                                default_initializer=nn.initializer.Constant(value=1),
                                                is_bias=False)
        bias = paddle.static.create_parameter(shape=[32],
                                              dtype='float32',
                                              is_bias=True) 
        self.vars.extend([weight, bias])
        running_mean = paddle.to_tensor(np.zeros([32], np.float32), stop_gradient=True)
        running_var = paddle.to_tensor(np.zeros([32], np.float32), stop_gradient=True)
        self.vars_bn.extend([running_mean, running_var])

        # ------------------------The third conv2d------------------------
        weight = paddle.static.create_parameter(shape=[32, 32, 7, 7],
                                                dtype='float32',
                                                default_initializer=nn.initializer.KaimingNormal(),  
                                                is_bias=False)
        bias = paddle.static.create_parameter(shape=[32],
                                              dtype='float32',
                                              is_bias=True)
        self.vars.extend([weight, bias])
        # The third BatchNorm
        weight = paddle.static.create_parameter(shape=[32],
                                                dtype='float32',
                                                default_initializer=nn.initializer.Constant(value=1),  
                                                is_bias=False)
        bias = paddle.static.create_parameter(shape=[32],
                                              dtype='float32',
                                              is_bias=True)  
        self.vars.extend([weight, bias])
        running_mean = paddle.to_tensor(np.zeros([32], np.float32), stop_gradient=True)
        running_var = paddle.to_tensor(np.zeros([32], np.float32), stop_gradient=True)
        self.vars_bn.extend([running_mean, running_var])


        # ------------------------FC Layer-----------------------
        weight = paddle.static.create_parameter(shape=[32 * 26* 19, 4],
                                                dtype='float32',
                                                default_initializer=nn.initializer.XavierNormal(),
                                                is_bias=False)
        bias = paddle.static.create_parameter(shape=[4],
                                              dtype='float32',
                                              is_bias=True)
        self.vars.extend([weight, bias])

    def forward(self, x, params=None, bn_training=True):

        if params is None:
            params = self.vars

        weight, bias = params[0], params[1]  
        x = F.conv2d(x, weight, bias, stride=1, padding=0)
        x = F.max_pool2d(x, 2, 2, 0) 
        weight, bias = params[2], params[3]  
        running_mean, running_var = self.vars_bn[0], self.vars_bn[1]
        x = F.batch_norm(x, running_mean, running_var, weight=weight, bias=bias, training=bn_training)
        x = F.relu(x)  

        weight, bias = params[4], params[5] 
        x = F.conv2d(x, weight, bias, stride=1, padding=0)
        x = F.max_pool2d(x, 2, 2, 0) 
        weight, bias = params[6], params[7] 
        running_mean, running_var = self.vars_bn[2], self.vars_bn[3]
        x = F.batch_norm(x, running_mean, running_var, weight=weight, bias=bias, training=bn_training)
        x = F.relu(x) 

        weight, bias = params[8], params[9] 
        x = F.conv2d(x, weight, bias, stride=1, padding=0)
        x = F.max_pool2d(x, 2, 2, 0)
        weight, bias = params[10], params[11]  
        running_mean, running_var = self.vars_bn[4], self.vars_bn[5]
        x = F.batch_norm(x, running_mean, running_var, weight=weight, bias=bias, training=bn_training)
        x = F.relu(x) 

        x = paddle.reshape(x, [x.shape[0], -1])  
        weight, bias = params[-2], params[-1] 
        x = F.linear(x, weight, bias)

        output = x

        return output

    def parameters(self, include_sublayers=True):
        return self.vars
        
