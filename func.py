import numpy as np


class Func:
    @staticmethod
    def conv_forward_naive(x, w, b, conv_param):
        """
        A naive implementation of the forward pass for a convolutional layer.

        The input consists of N data points, each with C channels, height H and
        width W. We convolve each input with F different filters, where each filter
        spans all C channels and has height HH and width WW.

        Input:
        - x: Input data of shape (N, C, H, W)
        - w: Filter weights of shape (F, C, HH, WW)
        - b: Biases, of shape (F,)
        - conv_param: A dictionary with the following keys:
          - 'stride': The number of pixels between adjacent receptive fields in the
            horizontal and vertical directions.
          - 'pad': The number of pixels that will be used to zero-pad the input.


        During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
        along the height and width axes of the input. Be careful not to modfiy the original
        input x directly.

        Returns a tuple of:
        - out: Output data, of shape (N, F, H', W') where H' and W' are given by
          H' = 1 + (H + 2 * pad - HH) / stride
          W' = 1 + (W + 2 * pad - WW) / stride

        - cache: (x, w, b, conv_param)
        """
        out = None
        ###########################################################################
        # TODO: Implement the convolutional forward pass.                         #
        # Hint: you can use the function np.pad for padding.                      #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        stride = conv_param['stride']
        pad = conv_param['pad']
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        H_r = 1 + (H + 2 * pad - HH) // stride
        W_r = 1 + (W + 2 * pad - WW) // stride
        xx = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')  # zero padding
        out = np.zeros((N, F, H_r, W_r))
        _, _, H_XX, W_XX = xx.shape
        for n in range(N):
            x_n = xx[n]
            for h_k in range(H_r):
                h_r = h_k * stride
                for w_k in range(W_r):
                    w_r = w_k * stride
                    xxx = x_n[:, h_r:h_r + HH, w_r:w_r + WW]
                    for f in range(F):
                        s = 0
                        for c in range(C):
                            s += np.sum(w[f, c] * xxx[c])
                        out[n][f][h_k][w_k] = s + b[f]

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        cache = (x, w, b, conv_param)
        return out, cache

    @staticmethod
    def conv_backward_naive(dout, cache):
        """
        A naive implementation of the backward pass for a convolutional layer.

        Inputs:
        - dout: Upstream derivatives.
        - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

        Returns a tuple of:
        - dx: Gradient with respect to x
        - dw: Gradient with respect to w
        - db: Gradient with respect to b
        """
        dx, dw, db = None, None, None
        ###########################################################################
        # TODO: Implement the convolutional backward pass.                        #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        x, w, b, conv_param = cache
        dw = np.zeros_like(w)
        stride = conv_param['stride']
        pad = conv_param['pad']
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        H_out = 1 + (H + 2 * pad - HH) // stride
        W_out = 1 + (W + 2 * pad - WW) // stride
        x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')  # zero padding
        dx_padded = np.zeros_like(x_padded)
        db = np.sum(dout, axis=(0, 2, 3))
        for h_out in range(H_out):
            for w_out in range(W_out):
                x_padded_slice = x_padded[:, :,
                                 h_out * stride: h_out * stride + HH,
                                 w_out * stride: w_out * stride + WW]  # 参与当前运算的图像切片
                dout_slice = dout[:, :, h_out, w_out]
                for f in range(F):
                    dw[f, :, :, :] += np.sum(x_padded_slice * (dout[:, f, h_out, w_out])[:, None, None, None], axis=0)
                for n in range(N):
                    dx_padded[n, :, h_out * stride:h_out * stride + HH, w_out * stride:w_out * stride + WW] += np.sum(
                        (w[:, :, :, :] * (dout[n, :, h_out, w_out])[:, None, None, None]), axis=0)

        # for n in range(N):
        #     x_n = x_padded[n]
        #     for h_out in range(H_out):
        #         h_r = h_out * stride
        #         for w_out in range(W_out):
        #             w_r = w_out * stride
        #             xxx = x_n[:, h_r:h_r + HH, w_r:w_r + WW]
        #             for f in range(F):
        #                 for c in range(C):
        #                     x_kernel_slice = x_padded[n, c, h_r:h_r + HH, w_r:w_r + WW]
        #                     dx_padded[n, c, h_r:h_r + HH, w_r:w_r + WW] += w[f, c]
        #                     # print(dw.shape, x_kernel_slice.shape)
        #                     dw[f, c] += x_kernel_slice
        dx = dx_padded[:, :, pad:-pad, pad:-pad]

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return dx, dw, db

    @staticmethod
    def max_pool_forward_naive(x, pool_param):
        """
        A naive implementation of the forward pass for a max-pooling layer.

        Inputs:
        - x: Input data, of shape (N, C, H, W)
        - pool_param: dictionary with the following keys:
          - 'pool_height': The height of each pooling region
          - 'pool_width': The width of each pooling region
          - 'stride': The distance between adjacent pooling regions

        No padding is necessary here. Output size is given by

        Returns a tuple of:
        - out: Output data, of shape (N, C, H', W') where H' and W' are given by
          H' = 1 + (H - pool_height) / stride
          W' = 1 + (W - pool_width) / stride
        - cache: (x, pool_param)
        """
        out = None
        ###########################################################################
        # TODO: Implement the max-pooling forward pass                            #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        N, C, H, W = x.shape
        pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
        H_out = 1 + (H - pool_height) // stride
        W_out = 1 + (W - pool_width) // stride
        out = np.zeros((N, C, H_out, W_out))
        for h_out in range(H_out):
            for w_out in range(W_out):
                xx = x[:, :, stride * h_out:stride * h_out + pool_height, stride * w_out:stride * w_out + pool_width]
                out[:, :, h_out, w_out] = np.max(xx, axis=(2, 3))
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        cache = (x, pool_param)
        return out, cache

    @staticmethod
    def max_pool_backward_naive(dout, cache):
        """
        A naive implementation of the backward pass for a max-pooling layer.

        Inputs:
        - dout: Upstream derivatives
        - cache: A tuple of (x, pool_param) as in the forward pass.

        Returns:
        - dx: Gradient with respect to x
        """
        dx = None
        ###########################################################################
        # TODO: Implement the max-pooling backward pass                           #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        x, pool_param = cache
        dx = np.zeros_like(x)
        N, C, H, W = x.shape
        _, _, H_out, W_out = dout.shape
        pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
        for h_out in range(H_out):
            for w_out in range(W_out):
                dx_slice = dx[:, :, stride * h_out:stride * h_out + pool_height,
                           stride * w_out:stride * w_out + pool_width]
                x_slice = x[:, :, stride * h_out:stride * h_out + pool_height,
                          stride * w_out:stride * w_out + pool_width]
                x_slice_max = np.max(x_slice, axis=(2, 3))
                x_mask = (x_slice == x_slice_max[:, :, None, None])
                dx_slice += (dout[:, :, h_out, w_out])[:, :, None, None] * x_mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return dx

    @staticmethod
    def affine_forward(x, w, b):
        """
        Computes the forward pass for an affine (fully-connected) layer.

        The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
        examples, where each example x[i] has shape (d_1, ..., d_k). We will
        reshape each input into a vector of dimension D = d_1 * ... * d_k, and
        then transform it to an output vector of dimension M.

        Inputs:
        - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
        - w: A numpy array of weights, of shape (D, M)
        - b: A numpy array of biases, of shape (M,)

        Returns a tuple of:
        - out: output, of shape (N, M)
        - cache: (x, w, b)
        """
        out = None
        ###########################################################################
        # TODO: Implement the affine forward pass. Store the result in out. You   #
        # will need to reshape the input into rows.                               #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        x1 = np.reshape(x, (np.size(x, 0), np.size(w, 0)))
        out = x1 @ w + b

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        cache = (x, w, b)
        return out, cache

    @staticmethod
    def affine_backward(dout, cache):
        """
        Computes the backward pass for an affine layer.

        Inputs:
        - dout: Upstream derivative, of shape (N, M)
        - cache: Tuple of:
          - x: Input data, of shape (N, d_1, ... d_k)
          - w: Weights, of shape (D, M)
          - b: Biases, of shape (M,)

        Returns a tuple of:
        - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
        - dw: Gradient with respect to w, of shape (D, M)
        - db: Gradient with respect to b, of shape (M,)
        """
        x, w, b = cache
        dx, dw, db = None, None, None
        ###########################################################################
        # TODO: Implement the affine backward pass.                               #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        dx = np.reshape(dout @ w.T, x.shape)
        N = x.shape[0]
        re_x = np.reshape(x, (N, -1))
        dw = re_x.T @ dout
        db = np.sum(dout, axis=0)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return dx, dw, db

    @staticmethod
    def relu_forward(x):
        """
        Computes the forward pass for a layer of rectified linear units (ReLUs).

        Input:
        - x: Inputs, of any shape

        Returns a tuple of:
        - out: Output, of the same shape as x
        - cache: x
        """
        out = None
        ###########################################################################
        # TODO: Implement the ReLU forward pass.                                  #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out = x * (x > 0)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        cache = x
        return out, cache

    @staticmethod
    def relu_backward(dout, cache):
        """
        Computes the backward pass for a layer of rectified linear units (ReLUs).

        Input:
        - dout: Upstream derivatives, of any shape
        - cache: Input x, of same shape as dout

        Returns:
        - dx: Gradient with respect to x
        """
        dx, x = None, cache
        ###########################################################################
        # TODO: Implement the ReLU backward pass.                                 #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        dx = dout * (x > 0)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return dx

