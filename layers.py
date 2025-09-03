import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import math
from utils import *


class ConvLayer(object):
    def __init__(self, input_size, num_channels, num_filters, batch_size, kernel_size, learning_rate, f, df, padding=0,
                 stride=1, device="cpu"):
        self.input_size = input_size
        self.num_channels = num_channels
        self.num_filters = num_filters
        self.batch_size = batch_size
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.output_size = math.floor((self.input_size + (2 * self.padding) - self.kernel_size) / self.stride) + 1
        self.learning_rate = learning_rate
        self.f = f
        self.df = df
        self.device = device
        self.kernel = torch.empty(self.num_filters, self.num_channels, self.kernel_size, self.kernel_size).normal_(
            mean=0, std=0.05).to(self.device)
        self.unfold = nn.Unfold(kernel_size=(self.kernel_size, self.kernel_size), padding=self.padding,
                                stride=self.stride).to(self.device)
        self.fold = nn.Fold(output_size=(self.input_size, self.input_size),
                            kernel_size=(self.kernel_size, self.kernel_size), padding=self.padding,
                            stride=self.stride).to(self.device)

    def forward(self, inp, network):
        self.X_col = self.unfold(inp.clone())
        self.flat_weights = self.kernel.reshape(self.num_filters, -1)

        M, K = self.flat_weights.shape
        adjustedXshape = self.X_col.squeeze(0)
        _, N = adjustedXshape.shape
        out = self.flat_weights @ self.X_col

        network.flop_count = network.flop_count + (2 * M * K * N)  # correct
        # print(network.flop_count)

        self.activations = out.reshape(self.batch_size, self.num_filters, self.output_size, self.output_size)
        return self.f(self.activations)

    def update_weights(self, e, network, update_weights=False, sign_reverse=False):
        fn_deriv = self.df(self.activations)
        num_elements = e.numel()  # = 1 × 124 × 124 × 124 = 1,907,776
        e = e * fn_deriv
        network.flop_count = network.flop_count + num_elements

        self.dout = e.reshape(self.batch_size, self.num_filters, -1)

        M, K = self.flat_weights.shape
        adjustedXshape = self.X_col.squeeze(0)
        _, N = adjustedXshape.shape
        dW = self.dout @ self.X_col.permute(0, 2, 1)
        network.flop_count = network.flop_count + (2 * M * K * N)

        # sums across batch dimension
        dW = torch.sum(dW, dim=0)

        dW = dW.reshape((self.num_filters, self.num_channels, self.kernel_size, self.kernel_size))
        if update_weights:
            if sign_reverse == True:  # This is necessary because PC and backprop learn gradients with different signs grad_pc = -grad_bp
                # print(dW.shape)

                # self.kernel -= self.learning_rate * torch.clamp(dW * 2,-50,50)
                mask = (dW < -0.00001) | (dW > 0.00001)
                # mask = dW != 0  # or (torch.abs(dW) > some_threshold)

                with torch.no_grad():
                    scaled_dw = dW * 2
                    clipped_dw = torch.clamp(scaled_dw, -50, 50)

                    # Only apply update where dW != 0
                    self.kernel[mask] -= self.learning_rate * clipped_dw[mask]
                    network.flop_count = network.flop_count + (3 * mask.sum().item())
            else:
                # self.kernel += self.learning_rate * torch.clamp(dW * 2,-50,50)
                mask = (dW < -0.00001) | (dW > 0.00001)
                # mask = dW != 0  # or (torch.abs(dW) > some_threshold)

                with torch.no_grad():
                    scaled_dw = dW * 2
                    clipped_dw = torch.clamp(scaled_dw, -50, 50)

                    # Only apply update where dW != 0
                    self.kernel[mask] += self.learning_rate * clipped_dw[mask]

                    network.flop_count = network.flop_count + (3 * mask.sum().item())
        return dW

    def backward(self, e, network):
        fn_deriv = self.df(self.activations)
        e = e * fn_deriv

        network.flop_count = network.flop_count + e.numel()
        self.dout = e.reshape(self.batch_size, self.num_filters, -1)
        M, K = self.flat_weights.shape
        adjustedDOUTshape = self.dout.squeeze(0)
        _, N = adjustedDOUTshape.shape
        dX_col = self.flat_weights.T @ self.dout
        network.flop_count = network.flop_count + (2 * M * K * N)
        dX = self.fold(dX_col)
        return torch.clamp(dX, -50, 50)

    def get_true_weight_grad(self):
        return self.kernel.grad

    def set_weight_parameters(self):
        self.kernel = nn.Parameter(self.kernel)

    def save_layer(self, logdir, i):
        np.save(logdir + "/layer_" + str(i) + "_weights.npy", self.kernel.detach().cpu().numpy())

    def load_layer(self, logdir, i):
        kernel = np.load(logdir + "/layer_" + str(i) + "_weights.npy")
        self.kernel = set_tensor(torch.from_numpy(kernel))


class MaxPool(object):
    def __init__(self, kernel_size, device='cpu'):
        self.kernel_size = kernel_size
        self.device = device
        self.activations = torch.empty(1)

    def forward(self, x, network):
        out, self.idxs = F.max_pool2d(x, self.kernel_size, return_indices=True)
        return out

    def backward(self, y, network):
        return F.max_unpool2d(y, self.idxs, self.kernel_size)

    def update_weights(self, e, network, update_weights=False, sign_reverse=False):
        return 0

    def get_true_weight_grad(self):
        return None

    def set_weight_parameters(self):
        pass

    def save_layer(self, logdir, i):
        pass

    def load_layer(self, logdir, i):
        pass


class AvgPool(object):
    def __init__(self, kernel_size, device='cpu'):
        self.kernel_size = kernel_size
        self.device = device
        self.activations = torch.empty(1)

    def forward(self, x):
        self.B_in, self.C_in, self.H_in, self.W_in = x.shape
        return F.avg_pool2d(x, self.kernel_size)

    def backward(self, y):
        N, C, H, W = y.shape
        return F.interpolate(y, scale_factor=(1, 1, self.kernel_size, self.kernel_size))

    def update_weights(self, e, update_weights=False, sign_reverse=False):
        return 0

    def save_layer(self, logdir, i):
        pass

    def load_layer(self, logdir, i):
        pass


class ProjectionLayer(object):
    def __init__(self, input_size, output_size, f, df, learning_rate, device='cpu'):
        self.input_size = input_size
        self.B, self.C, self.H, self.W = self.input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.f = f
        self.df = df
        self.device = device
        self.Hid = self.C * self.H * self.W
        self.weights = torch.empty((self.Hid, self.output_size)).normal_(mean=0.0, std=0.05).to(self.device)

    def forward(self, x, network):
        self.inp = x.detach().clone()
        out = x.reshape((len(x), -1))

        M, K = out.shape
        _, N = self.weights.shape

        self.activations = torch.matmul(out, self.weights)
        network.flop_count = network.flop_count + (2 * M * K * N)  # correct

        return self.f(self.activations)

    def backward(self, e, network):
        fn_deriv = self.df(self.activations)
        out = torch.matmul(e * fn_deriv, self.weights.T)
        # print(len(fn_deriv[0]))
        M, K = e.shape
        _, N = self.weights.T.shape

        out = out.reshape((len(e), self.C, self.H, self.W))
        network.flop_count = network.flop_count + (2 * M * K * N)  # correct
        return torch.clamp(out, -50, 50)

    def update_weights(self, e, network, update_weights=False, sign_reverse=False):
        out = self.inp.reshape((len(self.inp), -1))
        fn_deriv = self.df(self.activations)
        # print(e.shape)

        M, K = e.shape
        N, _ = out.T.shape

        dw = torch.matmul(out.T, e * fn_deriv)

        network.flop_count = network.flop_count + (M * K * N)  # correct

        if update_weights:
            if sign_reverse == True:

                mask = (dw < -0.00001) | (dw > 0.00001)

                # mask = dw != 0  # or (torch.abs(dW) > some_threshold)

                with torch.no_grad():
                    scaled_dw = dw * 2
                    clipped_dw = torch.clamp(scaled_dw, -50, 50)

                    # Only apply update where dW != 0
                    self.weights[mask] -= self.learning_rate * clipped_dw[mask]
                    network.flop_count = network.flop_count + (3 * mask.sum().item())

                # self.weights -= self.learning_rate * torch.clamp((dw * 2),-50,50)
            else:

                mask = (dw < -0.00001) | (dw > 0.00001)
                # mask = dw != 0  # or (torch.abs(dW) > some_threshold)

                with torch.no_grad():
                    scaled_dw = dw * 2
                    clipped_dw = torch.clamp(scaled_dw, -50, 50)

                    # Only apply update where dW != 0
                    self.weights[mask] += self.learning_rate * clipped_dw[mask]
                    network.flop_count = network.flop_count + (3 * mask.sum().item())  # correct

                # self.weights += self.learning_rate * torch.clamp((dw * 2),-50,50)
        return dw

    def get_true_weight_grad(self):
        return self.weights.grad

    def set_weight_parameters(self):
        self.weights = nn.Parameter(self.weights)

    def save_layer(self, logdir, i):
        np.save(logdir + "/layer_" + str(i) + "_weights.npy", self.weights.detach().cpu().numpy())

    def load_layer(self, logdir, i):
        weights = np.load(logdir + "/layer_" + str(i) + "_weights.npy")
        self.weights = set_tensor(torch.from_numpy(weights))


class FCLayer(object):
    def __init__(self, input_size, output_size, batch_size, learning_rate, f, df, device="cpu"):
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.f = f  # activation function
        self.df = df  # derivative of activation function, in this case relu_deriv
        self.device = device
        self.weights = torch.empty([self.input_size, self.output_size]).normal_(mean=0.0, std=0.05).to(self.device)

    def forward(self, x, network):
        self.inp = x.clone()  # inp = ([64, 3, 32, 32])
        self.activations = torch.matmul(self.inp, self.weights)  # neuron hi = w*inp in matrix

        M, K = self.inp.shape
        _, N = self.weights.shape

        network.flop_count = network.flop_count + (2 * M * K * N)  # correct
        return self.f(self.activations)  # f is the activation function

    def backward(self, e, network):
        self.fn_deriv = self.df(
            self.activations)  # df is derivative of activation function and activations is the pre-activation: xW+b
        torch.set_printoptions(threshold=float('inf'))  # Disable truncation
        out = torch.matmul(e * self.fn_deriv, self.weights.T)  # matrix multiplication (e*fn_deriv)*weights
        M, K = e.shape
        _, N = self.weights.T.shape

        network.flop_count = network.flop_count + (2 * M * K * N)  # correct

        return torch.clamp(out, -50, 50)  # clamp forces to have a boundary

    def update_weights(self, e, network, update_weights=False, sign_reverse=False):
        self.fn_deriv = self.df(self.activations)  # derivative with w*values
        M, K = e.shape
        N, _ = self.inp.T.shape
        # look over it

        dw = torch.matmul(self.inp.T, e * self.fn_deriv)

        network.flop_count = network.flop_count + (M * K * N)  # correct

        if update_weights:
            if sign_reverse == True:
                mask = (dw < -0.00001) | (dw > 0.00001)
                # mask = dw != 0  # or (torch.abs(dW) > some_threshold)

                with torch.no_grad():
                    scaled_dw = dw * 2
                    clipped_dw = torch.clamp(scaled_dw, -50, 50)

                    # Only apply update where dW != 0
                    self.weights[mask] -= self.learning_rate * clipped_dw[mask]

                    # Only update weights where the gradient is significant
                    network.flop_count = network.flop_count + (3 * mask.sum().item())

                # self.weights -= self.learning_rate * torch.clamp(dw*2,-50,50)
            else:
                mask = (dw < -0.00001) | (dw > 0.00001)

                # mask = dw != 0  # or (torch.abs(dW) > some_threshold)

                with torch.no_grad():
                    scaled_dw = dw * 2
                    clipped_dw = torch.clamp(scaled_dw, -50, 50)

                    # Only apply update where dW != 0
                    self.weights[mask] += self.learning_rate * clipped_dw[mask]
                    network.flop_count = network.flop_count + (3 * mask.sum().item())  # correct

                    # Mask out near-zero updates
                    # mask = (clipped_dw.abs() > 0.5)
                    # Show how many weights will be updated
                    # print(f"Number of significant updates: {mask.sum().item()}")

                # self.weights += self.learning_rate * torch.clamp(dw*2,-50,50)

        return dw

    def get_true_weight_grad(self):
        return self.weights.grad

    def set_weight_parameters(self):
        self.weights = nn.Parameter(self.weights)

    def save_layer(self, logdir, i):
        np.save(logdir + "/layer_" + str(i) + "_weights.npy", self.weights.detach().cpu().numpy())

    def load_layer(self, logdir, i):
        weights = np.load(logdir + "/layer_" + str(i) + "_weights.npy")
        self.weights = set_tensor(torch.from_numpy(weights))
