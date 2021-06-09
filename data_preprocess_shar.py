# encoding=utf-8
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from utils import get_sample_weights
from torchvision import transforms
import pickle as cp
from sliding_window import sliding_window
import scipy.io
from sklearn.model_selection import StratifiedShuffleSplit


def load_domain_data(domain_idx):
    """ to load all the data from the specific domain
    :param domain_idx:
    :return: X and y data of the entire domain
    """
    data_dir = './data/UniMiB-SHAR/'
    saved_filename = 'shar_domain_' + domain_idx + '_wd.data' # with domain label
    if os.path.isfile(data_dir + saved_filename) == True:
        data = np.load(data_dir + saved_filename, allow_pickle=True)
        X = data[0][0]
        y = data[0][1]
        d = data[0][2]
    else:
        str_folder = './data/UniMiB-SHAR/data/'
        data_all = scipy.io.loadmat(str_folder + 'acc_data.mat')
        y_id_all = scipy.io.loadmat(str_folder + 'acc_labels.mat')
        y_id_all = y_id_all['acc_labels']  # (11771, 3)

        X_all = data_all['acc_data']  # data: (11771, 453)
        y_all = y_id_all[:, 0] - 1  # to map the labels to [0, 16]
        id_all = y_id_all[:, 1]

        print('\nProcessing domain {0} files...\n'.format(domain_idx))

        target_idx = np.where(id_all == int(domain_idx))
        X = X_all[target_idx]
        y = y_all[target_idx]

        # domain index preprocessing
        domain_idx_now = int(domain_idx)
        if domain_idx_now < 4:
            domain_idx_int = domain_idx_now - 1
        elif domain_idx_now < 7 and domain_idx_now > 4:
            domain_idx_int = domain_idx_now - 2
        else:
            domain_idx_int = domain_idx_now - 4
        d = np.full(y.shape, domain_idx_int, dtype=int)

        print('\nProcessing domain {0} files | X: {1} y: {2} d:{3} \n'.format(domain_idx, X.shape, y.shape, d.shape))

        obj = [(X, y, d)]
        # file function is not supported in python3, use open instead
        f = open(os.path.join(data_dir, saved_filename), 'wb')
        cp.dump(obj, f, protocol=cp.HIGHEST_PROTOCOL)
        f.close()

    return X, y, d

class data_loader_shar(Dataset):
    def __init__(self, samples, labels, domains):
        self.samples = samples
        self.labels = labels
        self.domains = domains

    def __getitem__(self, index):
        sample, target, domain = self.samples[index], self.labels[index], self.domains[index]
        return sample, target, domain

    def __len__(self):
        return len(self.samples)


def prep_domains_shar(args, SLIDING_WINDOW_LEN=0, SLIDING_WINDOW_STEP=0):
    source_domain_list = ['1', '2', '3', '5']
    source_domain_list.remove(args.target_domain)

    # source domain data prep
    source_loaders = []
    for source_domain in source_domain_list:
        print('source_domain:', source_domain)
        x, y, d = load_domain_data(source_domain)
        x = x.reshape(-1, 151, 3)

        unique_y, counts_y = np.unique(y, return_counts=True)
        weights = 100.0 / torch.Tensor(counts_y)
        weights = weights.double()
        sample_weights = get_sample_weights(y, weights)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        data_set = data_loader_shar(x, y, d)
        source_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=sampler)
        print('source_loader batch: ', len(source_loader))
        source_loaders.append(source_loader)

    # target domain data prep
    print('target_domain:', args.target_domain)
    x, y, d = load_domain_data(args.target_domain)
    x = x.reshape(-1, 151, 3)
    data_set = data_loader_shar(x, y, d)
    target_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False)
    print('target_loader batch: ', len(target_loader))
    return source_loaders, target_loader



