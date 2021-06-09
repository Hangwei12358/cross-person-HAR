# encoding=utf-8
import matplotlib
matplotlib.use('Agg')

from train import train
from utils import set_name
import network_oppor as net
import data_preprocess_oppor
import torch
import argparse

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser(description='argument setting of network')
parser.add_argument('--now_model_name', type=str, default='GILE', help='the type of model')

parser.add_argument('--batch_size', type=int, default=64, help='batch size of training')
parser.add_argument('--n_epoch', type=int, default=100, help='number of training epochs')
parser.add_argument('--dataset', type=str, default='oppor', help='name of dataset')

parser.add_argument('--n_feature', type=int, default=77, help='name of feature dimension')
parser.add_argument('--len_sw', type=int, default=30, help='length of sliding window')
parser.add_argument('--n_class', type=int, default=18, help='number of class')
parser.add_argument('--d_AE', type=int, default=50, help='dim of AE')
parser.add_argument('--sigma', type=float, default=1, help='parameter of mmd')
parser.add_argument('--weight_mmd', type=float, default=1.0, help='weight of mmd loss')

parser.add_argument('--target_domain', type=str, default='S1', help='the target domain, [S1, S2, S3, S4]')
parser.add_argument('--test_every', type=int, default=1, help='do testing every n epochs')
parser.add_argument('-n_domains', type=int, default=4, help='number of total domains actually')
parser.add_argument('-n_target_domains', type=int, default=1, help='number of target domains')

parser.add_argument('--beta', type=float, default=1., help='multiplier for KL')

parser.add_argument('--x-dim', type=int, default=1152, help='input size after flattening')
parser.add_argument('--aux_loss_multiplier_y', type=float, default=1000., help='multiplier for y classifier')
parser.add_argument('--aux_loss_multiplier_d', type=float, default=1000., help='multiplier for d classifier')

parser.add_argument('--beta_d', type=float, default=0.002, help='multiplier for KL d')
parser.add_argument('--beta_x', type=float, default=0., help='multiplier for KL x')
parser.add_argument('--beta_y', type=float, default=10., help='multiplier for KL y')

parser.add_argument('--weight_true', type=float, default=1000.0, help='weights for classifier true')
parser.add_argument('--weight_false', type=float, default=1000.0, help='weights for classifier false')

if __name__ == '__main__':
    torch.manual_seed(10)
    args = parser.parse_args()
    args.device = DEVICE
    source_loaders, target_loader = data_preprocess_oppor.prep_domains_oppor(args, SLIDING_WINDOW_LEN=args.len_sw, SLIDING_WINDOW_STEP=int(0.5*args.len_sw))
    model = net.load_model(args)
    model = model.to(DEVICE)
    optimizer = net.set_up_optimizers(model.parameters())
    result_csv, result_txt, dir_name = set_name(args)
    train(model, DEVICE, optimizer, source_loaders, target_loader, result_csv, result_txt, args)

