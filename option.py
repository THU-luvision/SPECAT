import argparse


parser = argparse.ArgumentParser(description="Spatial-spEctral Cumulative-Attention Transformer Parameters")


# Hardware specifications
parser.add_argument("--gpu_id", type=str, default='0')

# Data specifications
parser.add_argument('--data_root', type=str, default='./dataset/', help='dataset directory')

# Saving specifications
parser.add_argument('--outf', type=str, default='./exp/', help='saving_path')

# Model specifications
parser.add_argument('--attention_type', type=str, default='full', help='attention type for ablation study, base, without MA or full')
parser.add_argument('--pretrained_model_path', type=str, default='./pre-trained/SPECAT_simu.pth', help='pretrained model directory')
# parser.add_argument('--pretrained_model_path', type=str, default=None, help='pretrained model directory')
parser.add_argument("--input_setting", type=str, default='H',
                    help='the input measurement of the network: H, HM or Y')
parser.add_argument("--input_mask", type=str, default='Mask',
                    help='the input mask of the network: Phi, Phi_PhiPhiT, Mask or None')  # Phi: shift_mask for CASSI system   Mask: mask for optical filters-based system

# Training specifications
parser.add_argument('--batch_size', type=int, default=2, help='the number of HSIs per batch')
parser.add_argument("--max_epoch", type=int, default=300, help='total epoch')
parser.add_argument("--scheduler", type=str, default='CosineAnnealingLR', help='MultiStepLR or CosineAnnealingLR')
parser.add_argument("--milestones", type=int, default=[50,100,150,200,250], help='milestones for MultiStepLR')
parser.add_argument("--gamma", type=float, default=0.5, help='learning rate decay for MultiStepLR')
parser.add_argument("--epoch_sam_num", type=int, default=5000, help='the number of samples per epoch')
parser.add_argument("--learning_rate", type=float, default=0.0004)

opt = parser.parse_args()


# dataset
opt.data_path = f"{opt.data_root}/cave_1024_28/"
opt.mask_path = f"{opt.data_root}/mask/"
opt.test_path = f"{opt.data_root}/TSA_simu_data/Truth/"

for arg in vars(opt):
    if vars(opt)[arg] == 'True':
        vars(opt)[arg] = True
    elif vars(opt)[arg] == 'False':
        vars(opt)[arg] = False
