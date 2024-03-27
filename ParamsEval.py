from SPECAT import SPECAT
from utils import my_summary

import torch
import torch.nn as nn
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def main():
    model = SPECAT(dim = 28, stage = 1, num_blocks = [2,1], attention_type='full').cuda()
    fc_layers = 0
    total_params = 0
    my_summary(model)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            fc_layers += 1
            total_params += sum(p.numel() for p in module.parameters())
    
    print('Full Connection Layers:',fc_layers)
    print('FCL params:',total_params)
    
if __name__ == '__main__':
    main()
