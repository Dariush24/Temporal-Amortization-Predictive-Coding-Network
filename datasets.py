import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import random
import numpy as np
import torchvision
from torch.utils.data import TensorDataset, SequentialSampler
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
from torchvision import datasets
from torchvision.io import read_image
from torchvision.transforms import ToTensor
from torchvision.datasets.folder import default_loader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torchvision import datasets
import torchvision.transforms.functional as TF

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from utils import *


def imshow(img):

    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def show_dataset(dataset):
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    images, labels = dataset[0]

    print("IMAGES: ", images.shape)
    print("LABELS: ", labels.shape)

    # print(onehot(labels, vocab_size=30))
    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


def get_cnn_dataset(dataset, batch_size):
    if dataset == "cifar":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR10(root='./cifar_data', train=True,
                                                download=True, transform=transform)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True)

        train_data = list(iter(trainloader))

        testset = torchvision.datasets.CIFAR10(root='./cifar_data', train=False,
                                               download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=True)
        test_data = list(iter(testloader))

        print(print("these is the shape ",test_data[0][0].shape))  # image shape


        print(test_data)

    elif dataset == "cifar100":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR100(root='./cifar100_data', train=True,
                                                 download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True)
        train_data = list(iter(trainloader))
        testset = torchvision.datasets.CIFAR100(root='./cifar100_data', train=False,
                                                download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=True)
        test_data = list(iter(testloader))

    elif dataset == "svhn":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.SVHN(root='./svhn_data', split='train',
                                             download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True)
        train_data = list(iter(trainloader))
        testset = torchvision.datasets.SVHN(root='./svhn_data', split='test',
                                            download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=True)
        test_data = list(iter(testloader))
    elif dataset == "mnist":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        mnist_transform = transforms.Compose([transforms.ToTensor(), mnist_normalize])
        trainset = torchvision.datasets.MNIST(root='./mnist_data', train=True,
                                              download=False, transform=mnist_transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True)
        train_data = list(iter(trainloader))
        testset = torchvision.datasets.MNIST(root='./mnist_data', train=False,
                                             download=False, transform=mnist_transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False)
        test_data = list(iter(testloader))
    elif dataset == "coil20":

        # data_dir = '/content/drive/MyDrive/coil-20-proc/coil-20-proc2' # Replace with your data path
        # print("asgasgdashgas")

        # Get a list of files
        # file_list = os.listdir(data_dir)

        # Sort the file list naturally
        # naturally_sorted_file_list = natsorted(file_list)

        # Usage:
        # data_dir = '/content/drive/MyDrive/coil-20-proc/coil-20-proc2'

        data_dir = r"C:\Users\dariu\Desktop\Studium\Vorlesungen und Uebungsblaetter\10.Semester\Bachelorthesis\dataset\coil-20-proc\coil-20-proc"

        transform = transforms.Compose([
            # transforms.Grayscale(),  # Single channel grayscale
            transforms.Grayscale(num_output_channels=1),  # Single channel grayscale
            transforms.Resize((128, 128)),  # Resize (if needed)
            transforms.ToTensor()  # Convert to tensor [0, 1]
        ])

        dataset = datasets.ImageFolder(root=data_dir, transform=transform)

        image, label = dataset[74]  # First image in the dataset
        print("Image shape:", image.shape)  # e.g., torch.Size([3, 256, 256])
        print("Label:", label)

        # dataset = datasets.ImageFolder(root=data_dir, transform=transform)

        # Optional: split into train/test
        # train_size = int(0.8 * len(dataset))
        # test_size = len(dataset) - train_size

        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)

        generator = torch.Generator().manual_seed(42)

        # train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size], generator=generator)
        # train_set, test_set = torch.utils.data.random_split(dataset, [0.8, 0.2], generator=generator)
        # train_set, val_set, test_set = torch.utils.data.random_split(dataset, [1, 0, 0], generator = generator)

        transform = transforms.Compose([transforms.ToTensor()])
        # dataset = datasets.ImageFolder(root="data/", transform=transform)
        dataset.samples.sort(key=lambda x: x[0])  # Sort by filename (optional)

        # Ordered split (no randomness)
        n = len(dataset)
        # train_set = Subset(dataset, range(0, int(0.8 * n)))
        train_set = dataset

        # Ordered DataLoader
        trainloader = DataLoader(train_set, batch_size=1, shuffle=False, num_workers=0, generator=generator)
        # Step 1: Convert to flat list (img, label)
        train_data = [(img, label.item()) for img, label in trainloader]

        # Step 2: Map image tensors to filepaths (relies on dataset.samples order matching DataLoader order)
        img_to_path = {}
        for i, (img, label) in enumerate(train_data):
            filepath, _ = train_set.samples[i]  # This gives full path
            img_to_path[img] = filepath

        # Step 3: Group by label
        buckets = defaultdict(list)
        for img, label in train_data:
            buckets[label].append(img)

        # Step 4: Helper to extract number from filename
        def extract_num(filepath):
            base = os.path.basename(filepath)  # obj3__17.png
            parts = base.split("__")
            if len(parts) < 2:
                return -1
            num_part = parts[-1].replace(".png", "")
            return int(num_part) if num_part.isdigit() else -1

        # Step 5: Sort inside each label bucket by extracted number
        for label in buckets:
            buckets[label].sort(key=lambda img: extract_num(img_to_path[img]))

        # Step 6: Flatten sorted data back into a list
        train_data_sorted = []
        for label in sorted(buckets.keys()):
            for img in buckets[label]:
                train_data_sorted.append((img, label))

        #train_data_wrapped = [(img, [label]) for img, label in train_data_sorted]

        train_data_tensor_labels = [[img, torch.tensor([label])] for img, label in train_data_sorted]


        fixed_train_dat = []
        subset_list_test = []

        #print(print(train_data_tensor_labels[0][0].shape))

        #print(train_data_tensor_labels)



        first_sample = train_data_tensor_labels[143]  # This is a list: [tensor(image), tensor(label)]

        # Extract the image tensor (handle list wrapping)
        image_tensor = first_sample[0]  # This is a list: [tensor(...)]
        image_tensor = image_tensor[0]  # Now it's the actual tensor

        # Remove batch dimension if present (e.g., [1, C, H, W] -> [C, H, W])
        if len(image_tensor.shape) == 4:
           image_tensor = image_tensor.squeeze(0)

        # Convert to NumPy and handle grayscale
        image_np = image_tensor.numpy()
        if len(image_np.shape) == 3 and image_np.shape[0] == 1:  # Grayscale
            image_np = image_np.squeeze(0)  # Shape: [H, W]

        # Display
        plt.imshow(image_np, cmap='gray')
        plt.title(f"Label: {first_sample[1]}")  # Show the label
        plt.axis('off')
        plt.show()


        # print ("list of subsets ",subset_list)
        print("Total files loaded by ImageFolder:", len(dataset))
    else:
        raise Exception("dataset: " + str(dataset) + " not supported")

    print("Setup data:")
    print("Train: ", len(train_data))
    # print("Test: ", len(test_data))
    # return train_data, val_data, test_data
    # return subset_list_train, subset_list_test
    return train_data_tensor_labels, subset_list_test


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


def get_lstm_dataset(seq_length, batch_size, buffer_size=10000):
    path_to_file = tf.keras.utils.get_file('shakespeare.txt',
                                           'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
    text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
    vocab = sorted(set(text))
    char2idx = {u: i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)
    text_as_int = np.array([char2idx[c] for c in text])
    examples_per_epoch = len(text) // (seq_length + 1)

    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

    sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)
    dataset = sequences.map(split_input_target)

    dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
    dataset = list(iter(dataset))
    # get dataset in right format
    vocab_size = len(vocab)
    return dataset, vocab_size, char2idx, idx2char
