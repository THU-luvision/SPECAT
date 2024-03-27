from SPECAT import SPECAT
from utils import *
import scipy.io as scio
import torch
import os
import numpy as np
from option import opt

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    raise Exception('NO GPU!')

# Intialize mask
mask3d_batch, input_mask = init_mask(opt.mask_path, opt.input_mask, 10)

if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

def test(model):
    test_data = LoadTest(opt.test_path)
    test_gt = test_data.cuda().float()
    input_meas = init_meas(test_gt, mask3d_batch, opt.input_setting)
    model.eval()
    psnr_list, ssim_list = [], []
    with torch.no_grad():
        model_out = model(input_meas, input_mask)
    for k in range(test_gt.shape[0]):
        psnr_val = torch_psnr(model_out[k, :, :, :], test_gt[k, :, :, :])
        ssim_val = torch_ssim(model_out[k, :, :, :], test_gt[k, :, :, :])
        psnr_list.append(psnr_val.detach().cpu().numpy())
        ssim_list.append(ssim_val.detach().cpu().numpy())  
    psnr_mean = np.mean(np.asarray(psnr_list))
    ssim_mean = np.mean(np.asarray(ssim_list))
    print(f'Testing psnr = {psnr_mean}, ssim = {ssim_mean}')
    pred = np.transpose(model_out.detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    truth = np.transpose(test_gt.cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    model.train()
    return pred, truth

def main():
    # model
    
    model = SPECAT(dim=28, stage=1, num_blocks=[2, 1], attention_type=opt.attention_type).cuda()

    if opt.pretrained_model_path is not None:
        print(f'load model from {opt.pretrained_model_path}')
        checkpoint = torch.load(opt.pretrained_model_path)
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()},
                                strict=True)
    pred, truth = test(model)
    name = opt.outf + 'Test_result_SPECAT.mat'
    print(f'Save reconstructed HSIs as {name}')
    scio.savemat(name, {'truth': truth, 'pred': pred})

if __name__ == '__main__':
    main()