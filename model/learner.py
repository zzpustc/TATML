import  paddle
from    paddle import nn
from    paddle.nn import functional as F
import paddle.fluid as fluid
import  numpy as np



class Learner(paddle.nn.Layer):
    """

    """

    def __init__(self, config, imgc, imgw, imgh):
        """

        :param config: network config file, type:list of (string, list)
        :param imgc: 1 or 3
        :param imgsz:  28 or 84
        """
        super(Learner, self).__init__()


        self.config = config

        # this dict contains all tensors needed to be optimized
        self.vars = []
        # running_mean and running_var
        self.vars_bn_mean = []
        self.vars_bn_var = []

        for i, (name, param) in enumerate(self.config):
            if name is 'conv2d':
                # [ch_out, ch_in, kernelsz, kernelsz]
                w = paddle.create_parameter(param[:4], dtype='float32', default_initializer=nn.initializer.KaimingNormal(),is_bias=False)
                # gain=1 according to cbfin's implementation
                self.vars.append(w)
                # [ch_out]
                b = paddle.create_parameter([param[0]], dtype='float32', is_bias=True)
                self.vars.append(b)

            elif name is 'convt2d':
                # [ch_in, ch_out, kernelsz, kernelsz, stride, padding]
                w = paddle.create_parameter(param[:4], dtype='float32', default_initializer=nn.initializer.KaimingNormal(),is_bias=False)
                # gain=1 according to cbfin's implementation
                self.vars.append(w)
                # [ch_in, ch_out]
                b = paddle.create_parameter([param[0]], dtype='float32', is_bias=True)
                self.vars.append(b)

            elif name is 'linear':
                # [ch_out, ch_in]
                w = paddle.create_parameter(param, dtype='float32', default_initializer=nn.initializer.KaimingNormal(),is_bias=False)
                # gain=1 according to cbfinn's implementation
                self.vars.append(w)
                # [ch_out]
                b = paddle.create_parameter([param[1]], dtype='float32', is_bias=True)
                self.vars.append(b)

            elif name is 'bn':
                # [ch_out]
                w = paddle.create_parameter([param[0]], dtype='float32', default_initializer=nn.initializer.Constant(value=1),is_bias=False)
                self.vars.append(w)
                # [ch_out]
                b = paddle.create_parameter([param[0]], dtype='float32',is_bias=True)
                self.vars.append(b)

                # must set requires_grad=False
                running_mean = paddle.to_tensor(np.zeros([param[0]], np.float32), stop_gradient = True)
                running_var = paddle.to_tensor(np.zeros([param[0]], np.float32), stop_gradient = True)
                self.vars_bn_mean.append(running_mean)
                self.vars_bn_var.append(running_var)


            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid']:
                continue
            else:
                raise NotImplementedError



    def forward(self, x, vars=None, bn_training=True):
        """
        This function can be called by finetunning, however, in finetunning, we dont wish to update
        running_mean/running_var. Thought weights/bias of bn is updated, it has been separated by fast_weights.
        Indeed, to not update running_mean/running_var, we need set update_bn_statistics=False
        but weight/bias will be updated and not dirty initial theta parameters via fast_weiths.
        :param x: [b, 1, 28, 28]
        :param vars:
        :param bn_training: set False to not update
        :return: x, loss, likelihood, kld
        """

        if vars is None:
            vars = self.vars

        idx = 0
        bn_idx = 0

        for name, param in self.config:
            if name is 'conv2d':
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name is 'convt2d':
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name is 'linear':
                w, b = vars[idx], vars[idx + 1]
                o = F.linear(x, w, b)
                idx += 2
                # print('forward:', idx, x.norm().item())
            elif name is 'bn':
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn_mean[bn_idx], self.vars_bn_var[bn_idx]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                idx += 2
                bn_idx += 1

            elif name is 'flatten':
                x = x.reshape(((x.shape)[0], -1))
            elif name is 'reshape':
                # [b, 8] => [b, 2, 2, 2]
                x = x.view(x.size(0), *param)
            elif name is 'relu':
                x = F.relu(x)
            elif name is 'leakyrelu':
                x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
            elif name is 'tanh':
                x = F.tanh(x)
            elif name is 'sigmoid':
                x = F.sigmoid(x)
            elif name is 'upsample':
                x = F.upsample_nearest(x, scale_factor=param[0])
            elif name is 'max_pool2d':
                x = F.max_pool2d(x, param[0], param[1], param[2])
            elif name is 'avg_pool2d':
                x = F.avg_pool2d(x, param[0], param[1], param[2])

            else:
                raise NotImplementedError

        # make sure variable is used properly
        assert idx == len(vars)
        assert bn_idx == len(self.vars_bn_mean)


        return o



    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars