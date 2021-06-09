# encoding=utf-8

import torch
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from model_GILE_ucihar import GILE

def load_model(args):
    if args.now_model_name == 'GILE':
        model = GILE(args)
    else:
        print('model not available!\n')
    return model

def set_up_optimizers(parameters):
    optimizer = torch.optim.Adam(parameters, lr=1e-4)
    return optimizer


