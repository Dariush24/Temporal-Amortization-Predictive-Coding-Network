import pickle
import shutil
import time
import matplotlib.pyplot as plt
import random
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import optuna
import torch
import torchvision
import torchvision.transforms as transforms
from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import time
import matplotlib.pyplot as plt
import torch.optim as optim
import subprocess
import sklearn.manifold
from sklearn.manifold import TSNE
import argparse
from datetime import datetime

from matplotlib import image as mpimg
from sklearn.model_selection import train_test_split
import datasource
from datasource import *
from utils import *
from layers import *
import seaborn as sns
import matplotlib.pyplot as plt


class PCNet(object):
    def __init__(self, layers, n_inference_steps_train, inference_learning_rate, loss_fn, loss_fn_deriv, device='cpu',
                 numerical_check=False, hidden_state=[], record_diffs=[], record_diffs_frame=[], count_updates=0,
                 flop_count=0, L2Class1=[]):
        self.layers = layers
        self.n_inference_steps_train = n_inference_steps_train
        self.inference_learning_rate = inference_learning_rate
        self.device = device
        self.loss_fn = loss_fn
        self.loss_fn_deriv = loss_fn_deriv
        self.L = len(self.layers)
        self.outs = [[] for i in range(self.L + 1)]
        self.prediction_errors = [[] for i in range(self.L + 1)]
        self.predictions = [[] for i in range(self.L + 1)]
        self.mus = [[] for i in range(self.L + 1)]
        self.numerical_check = numerical_check
        self.hidden_state = hidden_state
        self.record_diffs = record_diffs
        self.record_diffs_frame = record_diffs_frame
        self.count_updates = count_updates
        self.flop_count = flop_count
        self.L2Class1 = L2Class1

        if self.numerical_check:
            print("Numerical Check Activated!")
            for l in self.layers:
                l.set_weight_parameters()

    def count_nonzero_elements(self, dW, epsilon):
        if torch.is_tensor(dW):
            return (dW.abs() > epsilon).sum().item()
        elif isinstance(dW, (list, tuple)):
            return sum((w.abs() > epsilon).sum().item() for w in dW)
        else:
            return 0

    def update_weights(self, print_weight_grads=False, get_errors=False):
        weight_diffs = []
        count_not_near_zero = 0
        for (i, l) in enumerate(self.layers):
            if i != 1:
                if self.numerical_check:
                    true_weight_grad = l.get_true_weight_grad().clone()
                true_dW = l.update_weights(self.predictions[i + 1], self, update_weights=True)

                diff = torch.sum((0 - true_dW) ** 2).item()

                weight_diffs.append(diff)
                epsilon = 0.00001

                count_not_near_zero += self.count_nonzero_elements(true_dW, epsilon)

                # count_not_near_zero += self.count_nonzero_elements(true_dW, epsilon)

                """
                for i in range(len(dW)):                 
                    count_not_near_zero += torch.sum(dW[i].abs() > epsilon).item()
                """

                # print(self.count_updates)
                """
                if (self.layers[-1] == l):
                    print(dW[-1])
                    count_not_near_zero = torch.sum(dW[-1].abs() > epsilon).item()
                    print("Number of values not near zero:", count_not_near_zero)
                """

                if print_weight_grads:
                    print("weight grads : ", i)
                    print("dW: ", dW * 2)
                    print("true diffs: ", true_dW * 2)
                    if self.numerical_check:
                        print("true weights ", true_weight_grad)
        self.count_updates += count_not_near_zero
        return weight_diffs

    def forward(self, x, flop_count):
        for i, l in enumerate(self.layers):
            x = l.forward(x, self)
        return x

    def no_grad_forward(self, x):
        with torch.no_grad():
            for i, l in enumerate(self.layers):
                x = l.forward(x, self)
            return x

    def addFeatures(self, features, labelsList, dataset, testset, n_epochs, n_inference_steps,
                    savedir, logdir, old_savedir="", print_every=100, save_every=1):
        if old_savedir != "None":
            self.load_model(old_savedir)
        pcn_accuracies = []
        pcn_test_accuracies = []
        losses = []
        accs = []
        weight_diffs_list = []
        test_accs = []
        for epoch in range(n_epochs):
            losslist = []
            print("Epoch: ", epoch)
            for i, (inp, label) in enumerate(testset):
                # if self.loss_fn != cross_entropy_loss:
                # label = onehot(label).to(DEVICE)
                # else:
                labelsList.append(label)
                self.inferFeatures(inp.to(DEVICE), label, features)

    def inferFeatures(self, inp, label, features, n_inference_steps=None):
        self.n_inference_steps_train = n_inference_steps if n_inference_steps is not None else self.n_inference_steps_train
        with torch.no_grad():
            self.mus[0] = inp.clone()
            self.outs[0] = inp.clone()
            for i, l in enumerate(self.layers):
                # initialize mus with forward predictions
                self.mus[i + 1] = l.forward(self.mus[i])

            # print("these are the features: ", self.mus[-2])
            # print("these are the output: ", self.mus[-1])
            features.append(self.mus[-2])

    """
    def infer(self, inp, label,  number, epoch, n_inference_steps=None):
        self.n_inference_steps_train = n_inference_steps if n_inference_steps is not None else self.n_inference_steps_train
        with torch.no_grad():
          self.mus[0] = inp.clone()
          self.outs[0] = inp.clone()
          for i,l in enumerate(self.layers):
            #initialize mus with forward predictions
            self.mus[i+1] = l.forward(self.mus[i],self)
            self.outs[i+1] = self.mus[i+1].clone()
          self.mus[-1] = label.clone() #setup final label
          self.prediction_errors[-1] = -self.loss_fn_deriv(self.outs[-1], self.mus[-1])#self.mus[-1] - self.outs[-1] #setup final prediction errors
          self.predictions[-1] = self.prediction_errors[-1].clone()
          _,K = self.outs[-1].shape
          self.flop_count = self.flop_count + 2*K #correct
          for n in range(self.n_inference_steps_train):
          #reversed inference
            for j in reversed(range(len(self.layers))):
              if j != 0:
                self.prediction_errors[j] = self.mus[j] - self.outs[j]
                self.flop_count = self.flop_count + self.mus[j].numel()
                self.predictions[j] = self.layers[j].backward(self.prediction_errors[j+1],self)
                dx_l = self.prediction_errors[j] - self.predictions[j]
                self.flop_count = self.flop_count + self.prediction_errors[j].numel()
                self.flop_count = self.flop_count + (3*dx_l.numel()) #correct

                self.mus[j] -= self.inference_learning_rate * (2*dx_l)
          #update weights
          weight_diffs = self.update_weights()
          #get loss:
          L = self.loss_fn(self.outs[-1],self.mus[-1]).item()#torch.sum(self.prediction_errors[-1]**2).item()
          self.flop_count = self.flop_count + (3*self.outs[-1].numel()-1)
          #get accuracy
          acc = accuracy(self.no_grad_forward(inp),label)
          return L,acc,weight_diffs
    """

    def infer(self, inp, label, number, epoch, n_inference_steps=None):
        captureStateUpdates = []
        self.n_inference_steps_train = n_inference_steps if n_inference_steps is not None else self.n_inference_steps_train
        with torch.no_grad():
            capturingStateUpdates = []
            # if it is the first sample, do normal procedure
            if (number == 0):

                self.mus[0] = inp.clone()
                self.outs[0] = inp.clone()

                for i, l in enumerate(self.layers):
                    # initialize mus with forward predictions
                    self.mus[i + 1] = l.forward(self.mus[i], self)
                    self.outs[i + 1] = self.mus[i + 1].clone()
                self.mus[-1] = label.clone()  # setup final label

                self.prediction_errors[-1] = -self.loss_fn_deriv(self.outs[-1], self.mus[
                    -1])  # self.mus[-1] - self.outs[-1] #setup final prediction errors

                _, K = self.outs[-1].shape

                self.flop_count = self.flop_count + 2 * K  # correct

                self.predictions[-1] = self.prediction_errors[-1].clone()
                for n in range(self.n_inference_steps_train):
                    # reversed inference
                    for j in reversed(range(len(self.layers))):
                        if j != 0:
                            self.prediction_errors[j] = self.mus[j] - self.outs[j]
                            self.flop_count = self.flop_count + self.mus[j].numel()

                            self.predictions[j] = self.layers[j].backward(self.prediction_errors[j + 1], self)
                            dx_l = self.prediction_errors[j] - self.predictions[j]

                            self.flop_count = self.flop_count + self.prediction_errors[j].numel()

                            self.mus[j] -= self.inference_learning_rate * (2 * dx_l)

                            self.flop_count = self.flop_count + (3 * dx_l.numel())  # correct

                self.hidden_state = [mu.clone() for mu in self.mus]

                # update weights
                weight_diffs = self.update_weights()
                # get loss:
                L = self.loss_fn(self.outs[-1], self.mus[-1]).item()  # torch.sum(self.prediction_errors[-1]**2).item()
                self.flop_count = self.flop_count + (3 * self.outs[-1].numel() - 1)

                # get accuracy
                acc = accuracy(self.no_grad_forward(inp), label)
                return L, acc, weight_diffs


            else:

                self.mus[0] = inp.clone()
                self.outs[0] = self.mus[0].clone()

                # try this version if the two loops under that don't work but with lower inference rate

                for i, l in enumerate(self.layers):
                    # initialize mus with forward predictions
                    self.mus[i + 1] = l.forward(self.mus[i], self)
                    self.outs[i + 1] = self.mus[i + 1].clone()

                for i, l in enumerate(self.layers):
                    # initialize mus with hidden state
                    self.mus[i + 1] = self.hidden_state[i + 1].clone()

                self.mus[-1] = label.clone()  # setup final label

                self.prediction_errors[-1] = -self.loss_fn_deriv(self.outs[-1], self.mus[
                    -1])  # self.mus[-1] - self.outs[-1] #setup final prediction errors

                _, K = self.outs[-1].shape

                self.flop_count = self.flop_count + 2 * K

                self.predictions[-1] = self.prediction_errors[-1].clone()
                for n in range(self.n_inference_steps_train):
                    # reversed inference
                    for j in reversed(range(len(self.layers))):
                        if j != 0:
                            self.prediction_errors[j] = self.mus[j] - self.outs[j]

                            self.flop_count = self.flop_count + self.mus[j].numel()

                            self.predictions[j] = self.layers[j].backward(self.prediction_errors[j + 1], self)

                            # print("predictions ", torch.norm(self.predictions[-3]))

                            dx_l = self.prediction_errors[j] - self.predictions[j]

                            self.flop_count = self.flop_count + self.prediction_errors[j].numel()

                            # print("dx_l ", torch.norm(dx_l))

                            self.mus[j] -= self.inference_learning_rate * (2 * dx_l)
                            self.flop_count = self.flop_count + (3 * dx_l.numel())  # correct

                # update weights

                weight_diffs = self.update_weights()
                self.hidden_state = [mu.clone() for mu in self.mus]

                # get loss:
                L = self.loss_fn(self.outs[-1], self.mus[-1]).item()  # torch.sum(self.prediction_errors[-1]**2).item()
                self.flop_count = self.flop_count + (3 * self.outs[-1].numel() - 1)

                # get accuracy
                acc = accuracy(self.no_grad_forward(inp), label)
                return L, acc, weight_diffs

    def decay_latent_state(self, z, alpha=0.95):
        z = torch.as_tensor(z)  # safely converts list or numpy array
        # alpha close to 1 means slow decay, close to 0 means fast decay
        return alpha * z

    def test_accuracy(self, testset):
        accs = []
        for i, (inp, label) in enumerate(testset):
            pred_y = self.no_grad_forward(inp.to(DEVICE))
            acc = accuracy(pred_y, onehot(label).to(DEVICE))
            accs.append(acc)
        return np.mean(np.array(accs)), accs

    def train(self, capture_epoch_acc, PC_epoch_accuracies, PC_epoch_accuracies_test, dataset, valset, testset,
              n_epochs,
              n_inference_steps,
              logdir, savedir, old_savedir, save_every=1,
              print_every=10):
        if old_savedir != "None":
            self.load_model(old_savedir)
        pcn_accuracies = []
        pcn_test_accuracies = []
        losses = []
        accs = []
        weight_diffs_list = []
        val_accs = []
        test_accs = []

        """
        samples_per_class = 72

        subset_list_train = []
        subset_list_test = []

        for class_id in range(20):
            start = class_id * samples_per_class
            end = start + samples_per_class

            class_subset = dataset[start:end]
            subset_list_train.append(class_subset)
        """

        """
        num_tasks = 20
        classes_per_task = 1
        samples_per_class = 72
        test_split_ratio = 0.1

        subset_list_train = []
        subset_list_test = []

        for task_id in range(num_tasks):
            task_train = []
            task_test = []

            for class_offset in range(classes_per_task):
                class_id = task_id * classes_per_task + class_offset
                class_start = class_id * samples_per_class
                class_end = class_start + samples_per_class

                # Get data for this class
                class_data = dataset[class_start:class_end]

                # Compute test/train split
                # num_test = int(test_split_ratio * samples_per_class)
                num_test = 8
                class_test = class_data[:num_test]
                class_train = class_data[num_test:]

                task_train += class_train
                task_test += class_test

            subset_list_train.append(task_train)
            subset_list_test.append(task_test)
          """

        num_tasks = 20
        classes_per_task = 1
        samples_per_class = 72

        subset_list_train = []
        subset_list_test = []

        num_test = 8  # Number of test samples per class

        for task_id in range(num_tasks):
            task_train = []
            task_test = []

            for class_offset in range(classes_per_task):
                class_id = task_id * classes_per_task + class_offset
                class_start = class_id * samples_per_class
                class_end = class_start + samples_per_class

                # Get data for this class
                class_data = dataset[class_start:class_end]

                # Split: first samples as train set, last samples as test set
                class_train = class_data[:samples_per_class - num_test]  # Train: first samples
                class_test = class_data[samples_per_class - num_test:]  # Test: last samples

                # Append to task lists
                task_train += class_train
                task_test += class_test

            subset_list_train.append(task_train)
            subset_list_test.append(task_test)

        """
        num_tasks = 20
        classes_per_task = 1
        samples_per_class = 72

        num_test = round(samples_per_class * 0.10)  # ≈ 7
        num_val = round(samples_per_class * 0.10)  # ≈ 7

        subset_list_train = []
        subset_list_test = []

        for task_id in range(num_tasks):
            task_train = []
            task_test = []

            for class_offset in range(classes_per_task):
                class_id = task_id * classes_per_task + class_offset
                class_start = class_id * samples_per_class
                class_end = class_start + samples_per_class

                # Get full class data
                class_data = dataset[class_start:class_end]
                all_indices = list(range(samples_per_class))

                # Randomly choose validation indices
                val_indices = random.sample(all_indices, num_val)
                remaining_indices = list(set(all_indices) - set(val_indices))

                # Randomly choose test indices from remaining
                test_indices = random.sample(remaining_indices, num_test)
                test_indices.sort()  # Keep test set ordered

                # Remaining are for training (including the former val indices)
                train_indices = list(set(all_indices) - set(test_indices))

                # Sort train indices to preserve sequence
                train_indices.sort()

                # Assign samples
                class_train = [class_data[i] for i in train_indices]
                class_test = [class_data[i] for i in test_indices]

                task_train += class_train
                task_test += class_test

            subset_list_train.append(task_train)
            subset_list_test.append(task_test)        
        """

        record_L2 = []
        record_avg_update_per_framePC = []
        for epoch in range(n_epochs):
            weights_before_update = []
            weights_after_update = []

            for idx, layer in enumerate(self.layers):
                if hasattr(layer, "weights"):
                    weights_before_update.append(layer.weights.clone())
                    # print("these are the weights of the layer: ", layer.weights[0])

            losslist = []

            number_of_classes = 20

            print("(new) Epoch: ", epoch)

            record_L2_frame = []
            for j in range(number_of_classes):
                testset = testset + subset_list_test[j]

                for i, (inp, label) in enumerate(subset_list_train[j]):
                    weights_before_update_frame = []
                    weights_after_update_frame = []
                    for idx, layer in enumerate(self.layers):
                        if hasattr(layer, "weights"):
                            weights_before_update_frame.append(layer.weights.clone())
                            # print(f"Layer {idx} has weights.")
                            # print("these are the weights of the layer: ", layer.weights[0])

                    if self.loss_fn != cross_entropy_loss:
                        label = onehot(label).to(DEVICE)
                    else:
                        label = label.long().to(DEVICE)
                    L, acc, weight_diffs = self.infer(inp.to(DEVICE), label, i, epoch)
                    losslist.append(L)

                    """
                    for idx, layer in enumerate(self.layers):
                      if hasattr(layer, "weights"):                    
                        weights_after_update_frame.append(layer.weights.clone())

                    for i, _ in enumerate(weights_before_update_frame):
                      difference = weights_before_update_frame[i] - weights_after_update_frame[i]
                      #print("this is the difference ", torch.norm(difference[0]))
                      self.record_diffs_frame.append(difference)

                    x_tensor_frame = torch.cat([d.flatten() for d in self.record_diffs_frame])

                    record_L2_frame.append(torch.norm(x_tensor_frame, p = 2))
                    #print(record_L2_frame)  # debug line

                    self.record_diffs_frame.clear()
                    """
                self.saveL2frame(record_L2_frame, n_epochs)
                self.L2Class1.append(record_L2_frame.copy())

                # record_L2_frame.clear()
                self.hidden_state = []

                mean_acc, acclist = self.test_accuracy(subset_list_train[j])
                accs.append(mean_acc)
                mean_loss = np.mean(np.array(losslist))
                losses.append(mean_loss)

                # mean_val_acc,_ = self.test_accuracy(valset)
                # val_accs.append(mean_val_acc)

                mean_test_acc, _ = self.test_accuracy(testset)
                capture_epoch_acc.append(mean_test_acc)

                test_accs.append(mean_test_acc)
                weight_diffs_list.append(weight_diffs)
                # pcn_accuracies.append((mean_acc, mean_test_acc))
                pcn_accuracies.append(mean_acc)
                pcn_test_accuracies.append(mean_test_acc)
                if (j == 19):
                    # make sure the directory exists
                    os.makedirs('results', exist_ok=True)
                    # Save to file
                    with open('results/pcn_accuracies.pkl', 'wb') as f:
                        pickle.dump(pcn_accuracies, f)
                    with open('results/pcn_test_accuracies.pkl', 'wb') as f:
                        pickle.dump(pcn_test_accuracies, f)

                    print("ACCURACY: ", mean_acc)
                    PC_epoch_accuracies.append(mean_acc)
                    print("TEST ACCURACY: ", mean_test_acc)
                    PC_epoch_accuracies_test.append(mean_test_acc)
                    print("SAVING MODEL")
                    self.save_model(logdir, savedir, losses, accs, weight_diffs_list, test_accs)

            avg_update_frame = self.count_updates / 1280

            print(avg_update_frame)
            record_avg_update_per_framePC.append(avg_update_frame)
            self.count_updates = 0
            testset = []

            for idx, layer in enumerate(self.layers):
                if hasattr(layer, "weights"):
                    weights_after_update.append(layer.weights.clone())

            for i, _ in enumerate(weights_before_update):
                difference = weights_before_update[i] - weights_after_update[i]
                # print("this is the difference ", torch.norm(difference[0]))
                self.record_diffs.append(difference)

            x_tensor = torch.cat([d.flatten() for d in self.record_diffs])

            record_L2.append(torch.norm(x_tensor, p=2))

            self.record_diffs.clear()

            # print(record_L2)
            print("FLOPs: ", self.flop_count)

        self.saveL2(record_L2, n_epochs)
        FLOPsPC = self.flop_count
        print(FLOPsPC)
        # save FLOPs
        os.makedirs('FLOPs', exist_ok=True)
        with open('FLOPs/FLOPsPC.pkl', 'wb') as f:
            pickle.dump(FLOPsPC, f)
            # save average update between frames
        os.makedirs('updateAVG', exist_ok=True)
        with open('updateAVG/record_avg_update_per_framePC.pkl', 'wb') as f:
            pickle.dump(record_avg_update_per_framePC, f)

        self.saveL2frameAVG(self.L2Class1, n_epochs)

    def saveL2frame(self, record, n_epochs):
        l2_norms = [val.cpu().item() for val in record]
        os.makedirs('L2normFrame', exist_ok=True)
        # Save to file
        with open('L2normFrame/L2normFramePC.pkl', 'wb') as f:
            pickle.dump(l2_norms, f)

    def saveL2frameAVG(self, record, n_epochs):
        # Transpose the data to group values by index
        transposed = zip(*record)

        # Compute averages
        averages = [sum(group) / len(record) for group in transposed]
        l2_normsAVG = [val.cpu().item() for val in averages]
        os.makedirs('L2normFrameAVG', exist_ok=True)
        # Save to file
        with open('L2normFrameAVG/L2normFramePCAVG.pkl', 'wb') as f:
            pickle.dump(l2_normsAVG, f)

    def saveL2(self, record, n_epochs):
        l2_norms = [val.cpu().item() for val in record]
        os.makedirs('L2norm', exist_ok=True)
        # Save to file
        with open('L2norm/L2normPC.pkl', 'wb') as f:
            pickle.dump(l2_norms, f)

    def save_model(self, logdir, savedir, losses, accs, weight_diffs_list, test_accs):
        for i, l in enumerate(self.layers):
            l.save_layer(logdir, i)
        np.save(logdir + "/losses.npy", np.array(losses))
        np.save(logdir + "/accs.npy", np.array(accs))
        np.save(logdir + "/weight_diffs.npy", np.array(weight_diffs_list))
        np.save(logdir + "/test_accs.npy", np.array(test_accs))

        shutil.copytree(logdir, savedir, dirs_exist_ok=True)

        # subprocess.call(['rsync','--archive','--update','--compress','--progress',str(logdir) +"/",str(savedir)], shell = True)

        # print("Rsynced files from: " + str(logdir) + "/ " + " to" + str(savedir))
        now = datetime.now()
        current_time = str(now.strftime("%H:%M:%S"))
        subprocess.call(['echo', 'saved at time: ' + str(current_time)], shell=True)

    def load_model(self, old_savedir):
        for (i, l) in enumerate(self.layers):
            l.load_layer(old_savedir, i)


class Backprop_CNN(object):

    def __init__(self, layers, loss_fn, loss_fn_deriv, record_diffs=[], record_diffs_frame=[], count_updates=0,
                 flop_count=0, L2Class1=[]):
        self.layers = layers
        self.xs = [[] for i in range(len(self.layers) + 1)]
        self.e_ys = [[] for i in range(len(self.layers) + 1)]
        self.loss_fn = loss_fn
        self.loss_fn_deriv = loss_fn_deriv
        self.record_diffs = record_diffs
        self.record_diffs_frame = record_diffs_frame
        self.count_updates = count_updates
        self.flop_count = flop_count
        self.L2Class1 = L2Class1
        for l in self.layers:
            l.set_weight_parameters()

    def forward(self, inp):
        self.xs[0] = inp  # inp is a vector and has more samples ([64, 3, 32, 32])
        for i, l in enumerate(self.layers):
            self.xs[i + 1] = l.forward(self.xs[i], self)
        return self.xs[-1]  # last output after it went through the activation function (matrix)

    def backward(self, e_y):
        self.e_ys[-1] = e_y  # e_y = dErrortotal|dout
        for (i, l) in reversed(list(enumerate(self.layers))):
            self.e_ys[i] = l.backward(self.e_ys[i + 1], self)
        return self.e_ys[0]

    def update_weights(self, print_weight_grads=False, update_weight=False, sign_reverse=False):
        # weights_before_update = []
        # weights_after_update = []
        # capture weights before update
        """
        for idx, layer in enumerate(self.layers):
          if hasattr(layer, "weights"):
            weights_before_update.append(layer.weights.clone())
            #print(f"Layer {idx} has weights.")
            #print("these are the weights of the layer: ", layer.weights[0])
        """
        # print("these are weights before update ", (weights_before_update[1][0]))
        count_not_near_zero = 0

        for (i, l) in enumerate(self.layers):

            dW = l.update_weights(self.e_ys[i + 1], self, update_weights=update_weight, sign_reverse=sign_reverse)

            epsilon = 0.00001

            count_not_near_zero += self.count_nonzero_elements(dW, epsilon)

            """
            if torch.is_tensor(dW):
                for i in range(len(dW)):                 
                    count_not_near_zero += torch.sum(dW[i].abs() > epsilon).item()
            """
            if print_weight_grads:
                print("weight grads Backprop: ", i)
                print("dW Backprop: ", dW * 2)
                print("weight grad Backprop: ", l.get_true_weight_grad())

            # print(count_not_near_zero)

        self.count_updates += count_not_near_zero

        """
        #capture weights after update
        for idx, layer in enumerate(self.layers):
          if hasattr(layer, "weights"):
            weights_after_update.append(layer.weights.clone())
        c = weights_before_update[1] - weights_after_update[1]
        """

    def count_nonzero_elements(self, dW, epsilon):
        if torch.is_tensor(dW):
            return (dW.abs() > epsilon).sum().item()
        elif isinstance(dW, (list, tuple)):
            return sum((w.abs() > epsilon).sum().item() for w in dW)
        else:
            return 0

    def save_model(self, savedir, logdir, losses, accs, test_accs):
        for i, l in enumerate(self.layers):
            l.save_layer(logdir, i)
        np.save(logdir + "/losses.npy", np.array(losses))
        np.save(logdir + "/accs.npy", np.array(accs))
        np.save(logdir + "/test_accs.npy", np.array(test_accs))

        shutil.copytree(logdir, savedir, dirs_exist_ok=True)

        # subprocess.call(['rsync','--archive','--update','--compress','--progress',str(logdir) +"/",str(savedir)])
        # print("Rsynced files from: " + str(logdir) + "/ " + " to" + str(savedir))
        now = datetime.now()
        # current_time = str(now.strftime("%H:%M:%S"))
        # subprocess.call(['echo', 'saved at time: ' + str(current_time)], shell=True)

    def load_model(self, old_savedir):  # evtl self entfernen
        for (i, l) in enumerate(self.layers):
            l.load_layer(old_savedir, i)

    def test_accuracy(self, testset):
        accs = []
        for i, (inp, label) in enumerate(testset):
            pred_y = self.forward(inp.to(DEVICE))
            acc = accuracy(pred_y, onehot(label).to(DEVICE))
            accs.append(acc)
        return np.mean(np.array(accs)), accs

    def train(self, BP_epoch_accuracies, BP_epoch_accuracies_test, dataset, valset, testset, trainloader, n_epochs,
              n_inference_steps,
              savedir, logdir, old_savedir="", print_every=100, save_every=1):

        if old_savedir != "None":
            self.load_model(old_savedir)
        with torch.no_grad():
            bpn_accuracies = []
            bpn_test_accuracies = []
            accs = []
            losses = []
            val_accs = []
            test_accs = []

            num_tasks = 20
            classes_per_task = 1
            samples_per_class = 72
            test_split_ratio = 0.1

            subset_list_train = []
            subset_list_test = []

            num_test = 8  # Number of test samples per class

            for task_id in range(num_tasks):
                task_train = []
                task_test = []

                for class_offset in range(classes_per_task):
                    class_id = task_id * classes_per_task + class_offset
                    class_start = class_id * samples_per_class
                    class_end = class_start + samples_per_class

                    # Get data for this class
                    class_data = dataset[class_start:class_end]

                    # Split: first samples as train set, last samples as test set
                    class_train = class_data[:samples_per_class - num_test]  # Train: first samples
                    class_test = class_data[samples_per_class - num_test:]  # Test: last samples

                    # Append to task lists
                    task_train += class_train
                    task_test += class_test

                subset_list_train.append(task_train)
                subset_list_test.append(task_test)

            record_L2 = []
            record_avg_update_per_frameBP = []
            for n in range(n_epochs):
                # epoch_time = 0
                print("Epoch backprop: ", n)
                weights_before_update = []
                weights_after_update = []

                for idx, layer in enumerate(self.layers):
                    if hasattr(layer, "weights"):
                        # print(idx)
                        weights_before_update.append(layer.weights.clone())
                        # print(f"Layer {idx} has weights.")
                        # print("these are the weights of the layer: ", layer.weights[0])

                record_L2_frame = []
                for j in range(20):
                    testset = testset + subset_list_test[j]
                    start_time = time.time()
                    losslist = []

                    for (i, (inp, label)) in enumerate(
                            subset_list_train[j]):  # inp (single sample) mit seinem label, also sample einzeln

                        weights_before_update_frame = []
                        weights_after_update_frame = []
                        for idx, layer in enumerate(self.layers):
                            if hasattr(layer, "weights"):
                                weights_before_update_frame.append(layer.weights.clone())
                                # print(f"Layer {idx} has weights.")
                                # print("these are the weights of the layer: ", layer.weights[0])

                        out = self.forward(inp.to(DEVICE))  # output matrix

                        if self.loss_fn != cross_entropy_loss:
                            label = onehot(label).to(DEVICE)
                        else:
                            label = label.long().to(DEVICE)

                        e_y = self.loss_fn_deriv(out, label)  # e_y = dErrortotal|dout

                        _, K = out.shape

                        self.flop_count = self.flop_count + 2 * K  # correct

                        class_index = label.argmax(dim=1).item()

                        # e_y = out - label
                        self.backward(e_y)
                        self.update_weights(print_weight_grads=False, update_weight=True, sign_reverse=True)
                        # loss = torch.sum(e_y**2).item()
                        loss = self.loss_fn(out, label).item()  # no matrix, just a simple value

                        self.flop_count = self.flop_count + (3 * out.numel() - 1)  # correct

                        losslist.append(loss)
                        """
                        for idx, layer in enumerate(self.layers):
                          if hasattr(layer, "weights"):                    
                            weights_after_update_frame.append(layer.weights.clone())
                        for i, _ in enumerate(weights_before_update_frame):
                          difference = weights_before_update_frame[i] - weights_after_update_frame[i]
                          self.record_diffs_frame.append(difference)

                        x_tensor_frame = torch.cat([d.flatten() for d in self.record_diffs_frame])

                        record_L2_frame.append(torch.norm(x_tensor_frame, p = 2))
                        self.record_diffs_frame.clear()
                        """
                    self.saveL2frame(record_L2_frame, 1280)
                    self.L2Class1.append(record_L2_frame.copy())
                    # record_L2_frame.clear()
                    mean_acc, acclist = self.test_accuracy(subset_list_train[j])
                    accs.append(mean_acc)
                    mean_loss = np.mean(np.array(losslist))
                    losses.append(mean_loss)
                    # mean_val_acc, _ = self.test_accuracy(valset)
                    # val_accs.append(mean_val_acc)
                    mean_test_acc, _ = self.test_accuracy(testset)
                    test_accs.append(mean_test_acc)
                    if (j == 19):
                        bpn_accuracies.append(mean_acc)
                        bpn_test_accuracies.append(mean_test_acc)
                        os.makedirs('results', exist_ok=True)
                        # Save to file
                        with open('results/bpn_accuracies.pkl', 'wb') as f:
                            pickle.dump(bpn_accuracies, f)
                        with open('results/bpn_test_accuracies.pkl', 'wb') as f:
                            pickle.dump(bpn_test_accuracies, f)
                        end_time = time.time()

                        print("ACCURACY: ", mean_acc)
                        BP_epoch_accuracies.append(mean_acc)
                        BP_epoch_accuracies_test.append(mean_test_acc)
                        # print("VALIDATION ACCURACY: ", mean_val_acc)
                        print("TEST ACCURACY: ", mean_test_acc)
                        print("SAVING MODEL")
                        durationBP = end_time - start_time
                        # print(args.network_type, " Duration: ", durationBP)
                        # time = duration
                    # self.save_model(logdir, savedir, losses, accs, test_accs)

                record_L2_frame = []
                # reset testset after one epoch
                avg_update_frame = self.count_updates / 1280
                print(avg_update_frame)
                record_avg_update_per_frameBP.append(avg_update_frame)
                self.count_updates = 0
                testset = []
                valset = []

                """
                for idx, layer in enumerate(self.layers):
                  if hasattr(layer, "weights"):
                    weights_after_update.append(layer.weights.clone())

                for i, _ in enumerate(weights_before_update):
                  difference = weights_before_update[i] - weights_after_update[i]
                  #print("this is the difference ", torch.norm(difference[0]))
                  self.record_diffs.append(difference)

                x_tensor = torch.cat([d.flatten() for d in self.record_diffs])

                record_L2.append(torch.norm(x_tensor, p = 2))

                self.record_diffs.clear()
                #print(record_L2)

                #print(self.flop_count)    
                """

            self.saveL2frameAVG(self.L2Class1, n_epochs)

            self.saveL2(record_L2, n_epochs)
            FLOPsBP = self.flop_count
            # save FLOPs
            os.makedirs('FLOPs', exist_ok=True)
            with open('FLOPs/FLOPsBP.pkl', 'wb') as f:
                pickle.dump(FLOPsBP, f)
            # Save average update between frames
            os.makedirs('updateAVG', exist_ok=True)
            with open('updateAVG/record_avg_update_per_frameBP.pkl', 'wb') as f:
                pickle.dump(record_avg_update_per_frameBP, f)

            # plt.plot(range(1, n_epochs + 1), accs, 'g--', label='Backprop Train Accuracy')
            # plt.plot(range(1, n_epochs + 1), test_accs, 'g-', label='Backprop Test Accuracy')
            # plt.xlabel('Epoch')
            # plt.ylabel('Accuracy')
            # plt.title('Model Accuracy During Training')
            # plt.legend()
            # plt.grid(True)
            # plt.savefig("pipeline_training_accuracy.png", dpi=300, bbox_inches='tight')
            # plt.show()

    def saveL2frame(self, record, n_epochs):
        l2_norms = [val.cpu().item() for val in record]
        os.makedirs('L2normFrame', exist_ok=True)
        # Save to file
        with open('L2normFrame/L2normFrameBP.pkl', 'wb') as f:
            pickle.dump(l2_norms, f)

    def saveL2frameAVG(self, record, n_epochs):
        # Transpose the data to group values by index
        transposed = zip(*record)

        # Compute averages
        averages = [sum(group) / len(record) for group in transposed]
        l2_normsAVG = [val.cpu().item() for val in averages]
        os.makedirs('L2normFrameAVG', exist_ok=True)
        # Save to file
        with open('L2normFrameAVG/L2normFrameBPAVG.pkl', 'wb') as f:
            pickle.dump(l2_normsAVG, f)

    def saveL2(self, record, n_epochs):
        l2_norms = [val.cpu().item() for val in record]
        os.makedirs('L2norm', exist_ok=True)
        # Save to file
        with open('L2norm/L2normBP.pkl', 'wb') as f:
            pickle.dump(l2_norms, f)

    def addFeatures(self, features, labelsList, dataset, testset, n_epochs, n_inference_steps,
                    savedir, logdir, old_savedir="", print_every=100, save_every=1):
        if old_savedir != "None":
            self.load_model(old_savedir)
        with torch.no_grad():
            bpn_accuracies = []
            bpn_test_accuracies = []
            accs = []
            losses = []
            test_accs = []
            for n in range(n_epochs):
                # epoch_time = 0
                start_time = time.time()
                print("Epoch backprop: ", n)
                losslist = []
                for (i, (inp, label)) in enumerate(
                        testset):  # inp (single sample) mit seinem label, also sample einzeln
                    labelsList.append(label)
                    self.forwardFeature(inp.to(DEVICE), features)  # output matrix

    def forwardFeature(self, inp, features):
        self.xs[0] = inp  # inp is a vector and has more samples ([64, 3, 32, 32])
        for i, l in enumerate(self.layers):
            self.xs[i + 1] = l.forward(self.xs[i], self)
        # print("features ", self.xs[-2])
        features.append(self.xs[-2])
        return features  # last output after it went through the activation function (matrix)


if __name__ == '__main__':
    BP_all_accuracies = []
    BP_all_accuracies_test = []
    PC_all_accuracies = []
    PC_all_accuracies_test = []

    global DEVICE
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # DEVICE = torch.device("cpu")
    parser = argparse.ArgumentParser()
    print("Initialized")
    # parsing arguments
    parser.add_argument("--plot_type", type=str, default="normal")
    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--savedir", type=str, default="savedir")
    parser.add_argument("--BPlogdir", type=str, default="BPlogs")
    parser.add_argument("--BPsavedir", type=str, default="BPsavedir")
    parser.add_argument("--PClogdir", type=str, default="PClogs")
    parser.add_argument("--PCsavedir", type=str, default="PCsavedir")
    parser.add_argument("--Coillogdir", type=str, default="Coillogdir")
    parser.add_argument("--Coilsavedir", type=str, default="Coilsavedir")
    parser.add_argument("--batch_size", type=int, default=64)
    # parser.add_argument("--modified_net",type=int, default="False")
    parser.add_argument("--learning_rate", type=float, default=0.0005)
    parser.add_argument("--N_epochs", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--old_savedir", type=str, default="None")
    parser.add_argument("--n_inference_steps", type=int, default=100)
    parser.add_argument("--inference_learning_rate", type=float, default=0.1)
    parser.add_argument("--network_type", type=str, default="pc")
    parser.add_argument("--dataset", type=str, default="cifar")
    parser.add_argument("--loss_fn", type=str, default="mse")
    args = parser.parse_args()
    print("Args parsed")


    def create_sequences(data_list, sequence_length):
        sequences = []
        for i in range(0, len(data_list) - sequence_length + 1, sequence_length):
            sequence = data_list[i:i + sequence_length]
            sequences.append(sequence)
        return sequences


    if (args.plot_type == "normal"):
        BP_all_accuracies = []
        BP_all_accuracies_test = []
        PC_all_accuracies = []
        PC_all_accuracies_test = []
        seedNumber = 1
        for i in range(seedNumber):
            torch.manual_seed(i)
            np.random.seed(i)
            random.seed(i)

            BP_epoch_accuracies = []
            BP_epoch_accuracies_test = []

            PC_epoch_accuracies = []
            PC_epoch_accuracies_test = []
            capture_epoch_acc = []

            durationBP = 0

            if args.savedir:  # Checks if string is not empty
                os.makedirs(args.savedir, exist_ok=True)
                print(f"Created save directory at: {os.path.abspath(args.savedir)}")

            if args.logdir:
                os.makedirs(args.logdir, exist_ok=True)
                print(f"Created log directory at: {os.path.abspath(args.logdir)}")
            print("folders created")
            dataset, testset = datasource.get_cnn_dataset(args.dataset, args.batch_size)
            trainloader = []
            valset = []
            loss_fn, loss_fn_deriv = parse_loss_function(args.loss_fn)

            if args.dataset in ["cifar", "mnist", "svhn"]:
                output_size = 10
            if args.dataset == "cifar100":
                output_size = 100
            if args.dataset == "coil20":
                output_size = 20


            def onehot(x):
                z = torch.zeros([len(x), output_size])
                for i in range(len(x)):
                    z[i, x[i]] = 1
                return z.float().to(DEVICE)


            def one_epoch_plot(capture_epoch_acc):
                epochs = np.arange(1, 20 + 1)

                # Plot only the average line
                plt.figure(figsize=(8, 5))

                plt.plot(epochs, capture_epoch_acc, label=f'PCN Test Class Accuracy first epoch', color='green',
                         marker='o')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.title(f'Average Accuracy Across Seeds: {seedNumber}')
                plt.title(
                    f'Batch Size: {args.batch_size}, Inference Steps: {args.n_inference_steps}, learning_rate: {args.learning_rate}, inference_learning_rate {args.inference_learning_rate}, dataset {args.dataset}, seeds {seedNumber}')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig('average_accuracy.png', dpi=300, bbox_inches='tight')
                plt.show()


            # l1 = ConvLayer(32,3,6,64,5,args.learning_rate,relu,relu_deriv,device=DEVICE)
            # l2 = MaxPool(2,device=DEVICE)
            # l3 = ConvLayer(14,6,16,64,5,args.learning_rate,relu,relu_deriv,device=DEVICE)
            # l4 = ProjectionLayer((64,16,10,10),120,relu,relu_deriv,args.learning_rate,device=DEVICE)
            # l5 = FCLayer(120,84,64,args.learning_rate,relu,relu_deriv,device=DEVICE)
            # l6 = FCLayer(84,10,64,args.learning_rate,linear,linear_deriv,device=DEVICE)
            # layers =[l1,l2,l3,l4,l5,l6]

            # l1 = ConvLayer(128, 1, 6, 64, 5, args.learning_rate, relu, relu_deriv, device=DEVICE)
            # l2 = MaxPool(2, device=DEVICE)
            # l3 = ConvLayer(62, 6, 16, 64, 5, args.learning_rate, relu, relu_deriv, device=DEVICE)
            # l4 = ProjectionLayer((64, 16, 58, 58), 256, relu, relu_deriv, args.learning_rate, device=DEVICE)
            # l5 = FCLayer(256, 150, 64, args.learning_rate, relu, relu_deriv, device=DEVICE)

            # coil 20 layers
            """
            #OLD MODEL
            l1 = ConvLayer(128, 1, 36, args.batch_size, 5, args.learning_rate, relu, relu_deriv, device=DEVICE)
            l2 = MaxPool(2, device=DEVICE)
            l3 = ConvLayer(62, 36, 69, args.batch_size, 5, args.learning_rate, relu, relu_deriv, device=DEVICE)
            l4 = ProjectionLayer((args.batch_size, 69, 58, 58), 192, relu, relu_deriv, args.learning_rate,
                                 device=DEVICE)
            l5 = FCLayer(192, 128, args.batch_size, args.learning_rate, relu, relu_deriv, device=DEVICE)
            """

            l1 = ConvLayer(128, 1, 124, args.batch_size, 5, args.learning_rate, relu, relu_deriv, device=DEVICE)
            l2 = MaxPool(2, device=DEVICE)
            # l3 = ConvLayer (62, 124, 16, args.batch_size, 3, args.learning_rate, relu, relu_deriv, device=DEVICE)
            l3 = ProjectionLayer((args.batch_size, 124, 62, 62), 200, relu, relu_deriv, args.learning_rate,
                                 device=DEVICE)

            l4 = FCLayer(200, 128, args.batch_size, args.learning_rate, relu, relu_deriv, device=DEVICE)

            # l1 = ConvLayer(128, 1, 8, args.batch_size, 5, args.learning_rate, relu, relu_deriv, device=DEVICE)
            # l2 = MaxPool(2, device=DEVICE)
            # l4 = ProjectionLayer((args.batch_size, 8, 62, 62), 200, relu, relu_deriv, args.learning_rate, device=DEVICE)

            # l1 = ConvLayer(128, 1, 5, args.batch_size, 5, args.learning_rate, relu, relu_deriv, device=DEVICE)
            # l2 = MaxPool(2, device=DEVICE)
            # l3 = ConvLayer(62, 5, 5, args.batch_size, 5, args.learning_rate, relu, relu_deriv, device=DEVICE)
            # l4 = MaxPool(2, device=DEVICE)
            # l5 = ConvLayer(58, 5, 56, args.batch_size, 4, args.learning_rate, relu, relu_deriv, device=DEVICE)
            # l6 = MaxPool(2, device=DEVICE)
            # l7 = ProjectionLayer((args.batch_size, 56, 55, 55), 256, relu, relu_deriv, args.learning_rate, device=DEVICE)
            # l8 = FCLayer(256, 96, args.batch_size,args.learning_rate,relu,relu_deriv,device=DEVICE)
            # l = FCLayer(128, 128, args.batch_size, args.learning_rate, relu, relu_deriv, device=DEVICE)

            # cifar layers
            # l1 = ConvLayer(32, 3, 6, 64, 5, args.learning_rate, relu, relu_deriv, device=DEVICE)
            # l2 = MaxPool(2, device=DEVICE)
            # l3 = ConvLayer(14, 6, 16, 64, 5, args.learning_rate, relu, relu_deriv, device=DEVICE)
            # l4 = ProjectionLayer((64, 16, 10, 10), 200, relu, relu_deriv, args.learning_rate, device=DEVICE)
            # l5 = FCLayer(200, 150, 64, args.learning_rate, relu, relu_deriv, device=DEVICE)
            if args.loss_fn == "crossentropy":
                l6 = FCLayer(150, output_size, args.batch_size, args.learning_rate, softmax, linear_deriv,
                             device=DEVICE)
            else:
                l5 = FCLayer(128, output_size, args.batch_size, args.learning_rate, linear, linear_deriv, device=DEVICE)

                # l6 = FCLayer(128, output_size, args.batch_size, args.learning_rate, linear, linear_deriv, device=DEVICE)
                # l6 = FCLayer(150, output_size, 64, args.learning_rate, linear, linear_deriv, device=DEVICE)
                # layers = [l1, l2, l3, l4, l5, l6]
            layers = [l1, l2, l3, l4, l5]

            # layers =[l1,l2,l,l3,l4,l5]

            # layers =[l1,l2,l3,l4,l5,l6]
            # l1 = ConvLayer(32,3,20,64,4,args.learning_rate,tanh,tanh_deriv,device=DEVICE)
            # l2 = ConvLayer(29,20,50,64,5,args.learning_rate,tanh,tanh_deriv,device=DEVICE)
            # l3 = ConvLayer(25,50,50,64,5,args.learning_rate,tanh,tanh_deriv,stride=2,padding=1,device=DEVICE)
            # l4 = ConvLayer(12,50,5,64,3,args.learning_rate,tanh,tanh_deriv,stride=1,device=DEVICE)
            # l5 = ProjectionLayer((64,5,10,10),200,sigmoid,sigmoid_deriv,args.learning_rate,device=DEVICE)
            # l6 = FCLayer(200,100,64,args.learning_rate,linear,linear_deriv,device=DEVICE)
            # l7 = FCLayer(100,50,64,args.learning_rate,linear,linear_deriv,device=DEVICE)
            # l8 = FCLayer(50,output_size,64,args.learning_rate,linear,linear_deriv,device=DEVICE)
            # layers =[l1,l2,l3,l4,l5,l6,l7,l8]
            # Optuna objective function

            sequences = create_sequences(dataset, sequence_length=72)

            if args.network_type == "pc":
                net = PCNet(layers, args.n_inference_steps, args.inference_learning_rate, loss_fn=loss_fn,
                            loss_fn_deriv=loss_fn_deriv, device=DEVICE)
                net.train(capture_epoch_acc, PC_epoch_accuracies, PC_epoch_accuracies_test, dataset, valset, testset,
                          args.N_epochs,
                          args.n_inference_steps, args.savedir, args.logdir, "None", args.save_every, args.print_every)
            elif args.network_type == "backprop":
                net = Backprop_CNN(layers, loss_fn=loss_fn, loss_fn_deriv=loss_fn_deriv)
                net.train(BP_epoch_accuracies, BP_epoch_accuracies_test, dataset, valset, testset, trainloader,
                          args.N_epochs,
                          args.n_inference_steps, args.savedir, args.logdir, "None", args.save_every, args.print_every)
            else:
                raise Exception("Network type not recognised: must be one of 'backprop', 'pc'")

            BP_all_accuracies.append(BP_epoch_accuracies)
            BP_all_accuracies_test.append(BP_epoch_accuracies_test)

            PC_all_accuracies.append(PC_epoch_accuracies)
            PC_all_accuracies_test.append(PC_epoch_accuracies_test)
            # net.train(dataset[0:-2],testset[0:-2],args.N_epochs,args.n_inference_steps,args.Coilsavedir,args.Coillogdir,"None",args.save_every,args.print_every)

        # one_epoch_plot(capture_epoch_acc)

        BP_accuracies_array = np.array(BP_all_accuracies)  # Shape: (5, num_epochs)
        BP_accuracies_array_test = np.array(BP_all_accuracies_test)

        PC_accuracies_array = np.array(PC_all_accuracies)
        PC_accuracies_array_test = np.array(PC_all_accuracies_test)

        BP_mean_acc = np.mean(BP_accuracies_array, axis=0)
        BP_mean_acc_test = np.mean(BP_accuracies_array_test, axis=0)

        PC_mean_acc = np.mean(PC_accuracies_array, axis=0)
        PC_mean_acc_test = np.mean(PC_accuracies_array_test, axis=0)

        # epochs = np.arange(len(BP_mean_acc))  # e.g., [0, 1, 2, ..., num_epochs - 1]
        epochs = np.arange(1, args.N_epochs + 1)

        # Plot only the average line
        plt.figure(figsize=(8, 5))

        if (args.network_type == "backprop"):
            os.makedirs('resultsSeeds', exist_ok=True)
            # Save to file
            with open('resultsSeeds/BP_mean_acc.pkl', 'wb') as f:
                pickle.dump(BP_mean_acc, f)
            with open('resultsSeeds/BP_mean_acc_test.pkl', 'wb') as f:
                pickle.dump(BP_mean_acc_test, f)

        if (args.network_type == "pc"):
            os.makedirs('resultsSeeds', exist_ok=True)
            # Save to file
            with open('resultsSeeds/PC_mean_acc.pkl', 'wb') as f:
                pickle.dump(PC_mean_acc, f)
            print("Saving PC_mean_acc_test:", PC_mean_acc_test)
            with open('resultsSeeds/PC_mean_acc_test.pkl', 'wb') as f:
                pickle.dump(PC_mean_acc_test, f)

        if os.path.exists('resultsSeeds/BP_mean_acc.pkl'):
            with open('resultsSeeds/BP_mean_acc.pkl', 'rb') as f:
                BP_mean_acc = pickle.load(f)
            print("it is in arg.dataset ", BP_mean_acc)
            print(len(epochs), len(BP_mean_acc), len(BP_mean_acc_test))
            # plt.plot(epochs, BP_mean_acc, label='BPN Mean Accuracy', color='red')
            print("it was plotted")
            with open('resultsSeeds/BP_mean_acc_test.pkl', 'rb') as f:
                BP_mean_acc_test = pickle.load(f)
            # plt.plot(epochs, BP_mean_acc_test, label=f'BPN Test Accuracy', linestyle='--', color='red')
            plt.plot(epochs, BP_mean_acc_test, label=f'BPN Test Accuracy', color='red', marker='o')

        if os.path.exists('resultsSeeds/PC_mean_acc.pkl'):
            with open('resultsSeeds/PC_mean_acc.pkl', 'rb') as f:
                PC_mean_acc = pickle.load(f)
            # plt.plot(epochs, PC_mean_acc, label='PCN Mean Accuracy', color='green')
            with open('resultsSeeds/PC_mean_acc_test.pkl', 'rb') as f:
                PC_mean_acc_test = pickle.load(f)

            # plt.plot(epochs, PC_mean_acc_test, label=f'PCN Test Accuracy', linestyle='--', color='green')
            plt.plot(epochs, PC_mean_acc_test, label=f'PCN Test Accuracy', color='green', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'Average Accuracy Across Seeds: {seedNumber}')
        plt.title(
            f'Batch Size: {args.batch_size}, Inference Steps: {args.n_inference_steps}, learning_rate: {args.learning_rate}, inference_learning_rate {args.inference_learning_rate}, dataset {args.dataset}, seeds {seedNumber}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('average_accuracy.png', dpi=300, bbox_inches='tight')
        # plt.show()

        # showPlot(args.batch_size, args.n_inference_steps, args.learning_rate, args.inference_learning_rate, args.dataset, durationBP)



    elif (args.plot_type == "t-sne"):
        if args.savedir:  # Checks if string is not empty
            os.makedirs(args.savedir, exist_ok=True)
        print(f"Created save directory at: {os.path.abspath(args.savedir)}")

        if args.logdir:
            os.makedirs(args.logdir, exist_ok=True)
            print(f"Created log directory at: {os.path.abspath(args.logdir)}")
        print("folders created")
        dataset, valset, testset = datasetshabi.get_cnn_dataset(args.dataset, args.batch_size)
        loss_fn, loss_fn_deriv = parse_loss_function(args.loss_fn)

        if args.dataset in ["cifar", "mnist", "svhn"]:
            output_size = 10
        if args.dataset == "cifar100":
            output_size = 100
        if args.dataset == "coil20":
            output_size = 20


        def onehot(x):
            z = torch.zeros([len(x), output_size])
            for i in range(len(x)):
                z[i, x[i]] = 1
            return z.float().to(DEVICE)


        l1 = ConvLayer(128, 1, 36, args.batch_size, 5, args.learning_rate, relu, relu_deriv, device=DEVICE)
        l2 = MaxPool(2, device=DEVICE)
        l3 = ConvLayer(62, 36, 69, args.batch_size, 5, args.learning_rate, relu, relu_deriv, device=DEVICE)
        l4 = ProjectionLayer((args.batch_size, 69, 58, 58), 192, relu, relu_deriv, args.learning_rate, device=DEVICE)
        l5 = FCLayer(192, 128, args.batch_size, args.learning_rate, relu, relu_deriv, device=DEVICE)
        l6 = FCLayer(128, output_size, args.batch_size, args.learning_rate, linear, linear_deriv, device=DEVICE)
        layers = [l1, l2, l3, l4, l5, l6]

        labelsList = []
        features = []

        if (args.network_type == "backprop"):
            net = Backprop_CNN(layers, loss_fn=loss_fn, loss_fn_deriv=loss_fn_deriv)
        elif (args.network_type == "pc"):
            net = PCNet(layers, args.n_inference_steps, args.inference_learning_rate, loss_fn=loss_fn,
                        loss_fn_deriv=loss_fn_deriv, device=DEVICE)
        net.load_model("logs")
        net.addFeatures(features, labelsList, dataset[0:-2], testset[0:-2], 1,
                        args.n_inference_steps, args.savedir, args.logdir, "logs", args.save_every, args.print_every)

        tsne = TSNE(n_components=2)

        # features = torch.cat(features).numpy()
        # features = np.vstack(features)  # stacks into shape (N, feature_dim)

        features = torch.cat(features).numpy()  # ✅ ensures it's a NumPy array

        print(type(features), features.shape)

        labelsList = torch.cat(labelsList).numpy()

        print(type(labelsList), labelsList.shape)

        features_2d = tsne.fit_transform(features)
        np.set_printoptions(threshold=np.inf)  # Remove limit on printing
        print("these are the labels ", labelsList)
        labels = torch.tensor(labelsList).numpy()  # shape: (N,)

        plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap="tab20", s=10)
        plt.title("t-SNE on Extracted Features")
        plt.colorbar()
        plt.show()






