# SPECAT: SPatial-spEctral Cumulative-Attention Transformer for High-Resolution Hyperspectral Image Reconstruction (CVPR2024)
Zhiyang Yao, Shuyang Liu, Xiaoyun Yuan, Lu Fang  
Tsinghua University
![test](https://github.com/zhiyang96/SPECAT/blob/main/images/SPECAT.jpg   'The structure of SPECAT')  
SPECAT utilizes Cumulative-Attention Blocks (CABs) within an efficient hierarchical framework to extract features from non-local spatial-spectral details. Furthermore, it employs a projection-object Dual-domain Loss Function (DLF) to integrate the optical path constraint, a physical aspect often overlooked in current methodologies. Ultimately, SPECAT not only significantly enhances the reconstruction quality of spectral details but also breaks through the bottleneck of mutual restriction between the cost and accuracy in existing algorithms.

https://github.com/zhiyang96/SPECAT/assets/51588349/29d7b863-589f-4adb-9801-f06badcd071a


## Project Description
The SPECAT is mainly used for compressive spectral imaging systems based on optical filters (e.g. liquid crystal tunable filter-based HSI, metasurface HSI, Fabry-Pérot filters-based HSI), aiming to offer a reliable and effective reconstruction algorithm suitable for future on-chip HSI systems. This repository provides SPECAT model and its training and testing codes for applications in optical filter-based hyperspectral imaging systems.
![test](https://github.com/zhiyang96/SPECAT/blob/main/images/Optical_filter_based_HSI.jpg   'Optical_filter_based_HSI system')
The SPECAT is also applicable to CASSI systems. However, the default shallow structure is not optimal for CASSI systems, and the network depth and the number of CAB modules can be further increased to improve performance. The pre-trained SPECAT model and codes for the CASSI system both on simulated and real datasets will be uploaded soon.
![test](https://github.com/zhiyang96/SPECAT/blob/main/images/CASSI.jpg   'CASSI')

## Environment Requirement

pytorch version >= 2.1.0

python >= 3.10.0

## Project Structure

|--SPECAT  
> |-- dataset  
>  > |-- mask  
>  > |-- CAVE_512_28    
>  > |-- cave_1024_28  
>  > |-- KAIST_CVPR2021  
>  > |-- TSA_real_data     
>  > |-- TSA_simu_data    

> |-- option.py  
> |-- ssim_torch.py  
> |-- test.py  
> |-- train.py  
> |-- ultis.py  
> |-- SPECAT.py  
> |-- Readme.md  

## Data Preparation

We provide a 3D mask of optical filter-based HSI system (i.e. Fabry-Pérot filters-based HSI) in the folder "dataset/mask/". The publicly available hyperspectral images for training and testing can be obtained in the  “MST++” repository (https://github.com/caiyuanhao1998/MST-plus-plus/).

## Train

python train.py --outf xxx --data_root your/dataset/path

## Test

python test.py --outf xxx --data_root your/dataset/path

## Acknowledgement

Thanks a lot for the outstanding work and dedication from (https://github.com/caiyuanhao1998/MST). The code structure and datasets are borrowed from MST++ and SST (https://github.com/xintangjin/SST). We sincerely appreciate their contributions.




