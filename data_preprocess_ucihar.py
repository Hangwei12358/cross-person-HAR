# encoding=utf-8
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from sliding_window import sliding_window
from utils import get_sample_weights
import scipy.io
import pickle as cp
from sklearn.model_selection import StratifiedShuffleSplit


# This is for parsing the X data, you can ignore it if you do not need preprocessing
def format_data_x(datafile):
    x_data = None
    for item in datafile:
        item_data = np.loadtxt(item, dtype=np.float)
        if x_data is None:
            x_data = np.zeros((len(item_data), 1))
        x_data = np.hstack((x_data, item_data))
    x_data = x_data[:, 1:]
    print('x_data.shape:', x_data.shape)
    X = None
    for i in range(len(x_data)):
        row = np.asarray(x_data[i, :])
        row = row.reshape(9, 128).T
        if X is None:
            X = np.zeros((len(x_data), 128, 9))
        X[i] = row
    print('X.shape:', X.shape)
    return X

def format_data_y(datafile):
    data = np.loadtxt(datafile, dtype=np.int) - 1
    return data


def load_domain_data(domain_idx):
    """ to load all the data from the specific domain
    :param domain_idx:
    :return: X and y data of the entire domain
    """
    data_dir = './data/ucihar/'
    saved_filename = 'ucihar_domain_' + domain_idx + '_wd.data' # with domain label

    if os.path.isfile(data_dir + saved_filename) == True:
        data = np.load(data_dir + saved_filename, allow_pickle=True)
        X = data[0][0]
        y = data[0][1]
        d = data[0][2]
    else:
        str_folder = './data/ucihar/'
        INPUT_SIGNAL_TYPES = [
            "body_acc_x_",
            "body_acc_y_",
            "body_acc_z_",
            "body_gyro_x_",
            "body_gyro_y_",
            "body_gyro_z_",
            "total_acc_x_",
            "total_acc_y_",
            "total_acc_z_"
        ]
        str_train_files = [str_folder + 'train/' + 'InertialSignals/' + item + 'train.txt' for item in
                           INPUT_SIGNAL_TYPES]
        str_test_files = [str_folder + 'test/' + 'InertialSignals/' + item + 'test.txt' for item in INPUT_SIGNAL_TYPES]
        str_train_y = str_folder + 'train/y_train.txt'
        str_test_y = str_folder + 'test/y_test.txt'

        str_train_id = str_folder + 'train/subject_train.txt'
        str_test_id = str_folder + 'test/subject_test.txt'

        X_train = format_data_x(str_train_files)
        X_test = format_data_x(str_test_files)
        Y_train = format_data_y(str_train_y)
        Y_test = format_data_y(str_test_y)
        id_train = format_data_y(
            str_train_id)  # origin: array([ 0,  2,  4,  5,  6,  7, 10, 13, 14, 15, 16, 18, 20, 21, 22, 24, 25, 26, 27, 28, 29])
        id_test = format_data_y(str_test_id)  # origin: array([ 1,  3,  8,  9, 11, 12, 17, 19, 23])

        X_all = np.concatenate((X_train, X_test), axis=0)
        y_all = np.concatenate((Y_train, Y_test), axis=0)
        id_all = np.concatenate((id_train, id_test), axis=0)

        print('\nProcessing domain {0} files...\n'.format(domain_idx))
        target_idx = np.where(id_all == int(domain_idx))
        X = X_all[target_idx]
        y = y_all[target_idx]
        d = np.full(y.shape, int(domain_idx), dtype=int)
        print('\nProcessing domain {0} files | X: {1} y: {2} d:{3} \n'.format(domain_idx, X.shape, y.shape, d.shape))

        obj = [(X, y, d)]
        # file function is not supported in python3, use open instead
        f = open(os.path.join(data_dir, saved_filename), 'wb')
        cp.dump(obj, f, protocol=cp.HIGHEST_PROTOCOL)
        f.close()

    return X, y, d

class data_loader_ucihar(Dataset):
    def __init__(self, samples, labels, domains, t):
        self.samples = samples
        self.labels = labels
        self.domains = domains
        self.T = t

    def __getitem__(self, index):
        sample, target, domain = self.samples[index], self.labels[index], self.domains[index]
        sample = self.T(sample)
        return np.transpose(sample, (1, 0, 2)), target, domain

    def __len__(self):
        return len(self.samples)


def prep_domains_ucihar(args, SLIDING_WINDOW_LEN=0, SLIDING_WINDOW_STEP=0):

    source_domain_list = ['0', '1', '2', '3', '4']
    source_domain_list.remove(args.target_domain)

    # source domain data prep
    source_loaders = []
    for source_domain in source_domain_list:
        print('source_domain:', source_domain)
        x, y, d = load_domain_data(source_domain)
        x = np.transpose(x.reshape((-1, 1, 128, 9)), (0, 2, 1, 3))
        unique_y, counts_y = np.unique(y, return_counts=True)
        weights = 100.0 / torch.Tensor(counts_y)
        weights = weights.double()
        sample_weights = get_sample_weights(y, weights)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0, 0, 0, 0, 0, 0, 0, 0, 0), std=(1, 1, 1, 1, 1, 1, 1, 1, 1))
        ])
        data_set = data_loader_ucihar(x, y, d, transform)
        source_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=sampler)
        print('source_loader batch: ', len(source_loader))
        source_loaders.append(source_loader)

    # target domain data prep
    print('target_domain:', args.target_domain)
    x, y, d = load_domain_data(args.target_domain)
    x = np.transpose(x.reshape((-1, 1, 128, 9)), (0, 2, 1, 3))
    data_set = data_loader_ucihar(x, y, d, transform)
    target_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False)
    print('target_loader batch: ', len(target_loader))
    return source_loaders, target_loader

