import numpy as np
import func


class Conv2d:
    """
    weight : (F,C,H,W)
    """

    def __init__(self, in_channel: int, out_channel: int, kernel_size: int, stride: int = 1, padding: int = 0,
                 bias=True):
        self.weight = np.zeros((out_channel, in_channel, kernel_size, kernel_size))
        self.bias = np.zeros(out_channel) if bias else None
        self.conv_param = {'stride': stride, "pad": padding}
        self.x = None
        self.dcache = None
        self.parameters = [self.weight , self.bias]

    def forward(self, x):
        self.x = x
        func.Func.conv_forward_naive(x, self.weight, self.bias, self.conv_param)

    def backprop(self, dout):
        cache = (self.x, self.weight, self.bias, self.conv_param)
        self.dcache.Func.conv_backward_naive(dout, cache)
    def step(self, optim):
