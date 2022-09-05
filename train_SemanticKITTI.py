#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
import argparse
import sys
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from network.BEV_Unet import BEV_Unet
from network.ptBEV import ptBEVnet
from dataloader.dataset import collate_fn_BEV,SemKITTI,SemKITTI_label_name,spherical_dataset,voxel_dataset
from network.lovasz_losses import lovasz_softmax
#ignore weird np warning
import warnings
warnings.filterwarnings("ignore")


def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    bin_count=np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2)
    return bin_count[:n ** 2].reshape(n, n)

def per_class_iu(hist):
    print(hist)
    return np.diag(hist)/(hist.sum(1) + hist.sum(0) - np.diag(hist))

def fast_hist_crop(output, target, unique_label):
    hist = fast_hist(output.flatten(), target.flatten(), np.max(unique_label)+1)
    hist=hist[unique_label,:]
    hist=hist[:,unique_label]
    return hist

def SemKITTI2train(label):
    if isinstance(label, list): #isinstance判断label是不是一个list
        return [SemKITTI2train_single(a) for a in label]
    else:
        return SemKITTI2train_single(label)

def SemKITTI2train_single(label):
    return label - 1 # uint8 trick

def main(args):
    #1、导入参数
    data_path = args.data_dir
    train_batch_size = args.train_batch_size
    val_batch_size = args.val_batch_size
    check_iter = args.check_iter
    model_save_path = args.model_save_path
    compression_model = args.grid_size[2]
    grid_size = args.grid_size
    pytorch_device = torch.device('cuda:0')
    model = args.model
    if model == 'polar':
        fea_dim = 9
        circular_padding = True
    elif model == 'traditional':
        fea_dim = 7
        circular_padding = False

    #2、prepare miou fun
    unique_label=np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    unique_label_str=[SemKITTI_label_name[x] for x in unique_label+1]

    #3、prepare model BEV_Unet ptBEVnet
    my_BEV_model=BEV_Unet(n_class=len(unique_label), n_height = compression_model, input_batch_norm = True, dropout = 0.5, circular_padding = circular_padding)
    my_model = ptBEVnet(my_BEV_model, pt_model = 'pointnet', grid_size =  grid_size, fea_dim = fea_dim, max_pt_per_encode = 256,
                            out_pt_fea_dim = 512, kernal_size = 1, pt_selection = 'random', fea_compre = compression_model)
    if os.path.exists(model_save_path):
        my_model.load_state_dict(torch.load(model_save_path))
    my_model.to(pytorch_device)

    optimizer = optim.Adam(my_model.parameters())
    loss_fun = torch.nn.CrossEntropyLoss(ignore_index=255)

    # 4、prepare dataset
    #导入KITTI数据
    train_pt_dataset = SemKITTI(data_path + '/sequences/', imageset = 'train', return_ref = True)
    val_pt_dataset = SemKITTI(data_path + '/sequences/', imageset = 'val', return_ref = True)
    
    #转换原始数据
    if model == 'polar': #环状卷积
        train_dataset=spherical_dataset(train_pt_dataset, grid_size = grid_size, flip_aug = True, ignore_label = 0,rotate_aug = True, fixed_volume_space = True)
        val_dataset=spherical_dataset(val_pt_dataset, grid_size = grid_size, ignore_label = 0, fixed_volume_space = True)
        
    elif model == 'traditional': #体素卷积
        train_dataset=voxel_dataset(train_pt_dataset, grid_size = grid_size, flip_aug = True, ignore_label = 0,rotate_aug = True, fixed_volume_space = True)
        val_dataset=voxel_dataset(val_pt_dataset, grid_size = grid_size, ignore_label = 0, fixed_volume_space = True)
    # 将输入数据按照batch_size封装成Tensor
    train_dataset_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                                    batch_size = train_batch_size,
                                                    collate_fn = collate_fn_BEV,
                                                    shuffle = True,
                                                    num_workers = 4)
    val_dataset_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                                    batch_size = val_batch_size,
                                                    collate_fn = collate_fn_BEV,
                                                    shuffle = False,
                                                    num_workers = 4)

    # training
    epoch=0
    best_val_miou=0
    start_training=False
    my_model.train() #模型训练
    global_iter = 0
    exce_counter = 0

    while True:
        loss_list=[]
        pbar = tqdm(total=len(train_dataset_loader)) #tqdm进度条
        for i_iter,(_,train_vox_label,train_grid,_,train_pt_fea) in enumerate(train_dataset_loader): #enumerate遍历所有变量
            # validation
            if global_iter % check_iter == 0:
                my_model.eval() #模型验证
                hist_list = []
                val_loss_list = []
                
                with torch.no_grad(): #而with torch.no_grad()则主要是用于停止autograd模块的工作，以起到加速和节省显存的作用
                    #导入验证数据集
                    for i_iter_val,(_,val_vox_label,val_grid,val_pt_labs,val_pt_fea) in enumerate(val_dataset_loader):
                        val_vox_label = SemKITTI2train(val_vox_label)
                        val_pt_labs = SemKITTI2train(val_pt_labs)
                        val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in val_pt_fea]
                        val_grid_ten = [torch.from_numpy(i[:,:2]).to(pytorch_device) for i in val_grid]
                        val_label_tensor=val_vox_label.type(torch.LongTensor).to(pytorch_device)

                        predict_labels = my_model(val_pt_fea_ten, val_grid_ten)
                        loss = lovasz_softmax(torch.nn.functional.softmax(predict_labels).detach(), val_label_tensor,ignore=255)+loss_fun(predict_labels.detach(),val_label_tensor)
                        predict_labels = torch.argmax(predict_labels,dim=1)
                        predict_labels = predict_labels.cpu().detach().numpy()
                        for count,i_val_grid in enumerate(val_grid):
                            hist_list.append(fast_hist_crop(predict_labels[count,val_grid[count][:,0],val_grid[count][:,1],val_grid[count][:,2]],val_pt_labs[count],unique_label))
                        val_loss_list.append(loss.detach().cpu().numpy())
                my_model.train()
                iou = per_class_iu(sum(hist_list))
                print('Validation per class iou: ')
                for class_name, class_iou in zip(unique_label_str,iou):
                    print('%s : %.2f%%' % (class_name, class_iou*100))
                val_miou = np.nanmean(iou) * 100
                del val_vox_label,val_grid,val_pt_fea,val_grid_ten
                
                # save model if performance is improved
                if best_val_miou<val_miou:
                    best_val_miou=val_miou
                    torch.save(my_model.state_dict(), model_save_path)

                print('Current val miou is %.3f while the best val miou is %.3f' %
                    (val_miou,best_val_miou))
                print('Current val loss is %.3f' %
                    (np.mean(val_loss_list)))
                if start_training:
                    print('epoch %d iter %5d, loss: %.3f\n' %
                        (epoch, i_iter, np.mean(loss_list)))
                print('%d exceptions encountered during last training\n' %
                    exce_counter)
                exce_counter = 0
                loss_list = []

            # training
            try:
                train_vox_label = SemKITTI2train(train_vox_label)
                train_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in train_pt_fea]
                train_grid_ten = [torch.from_numpy(i[:,:2]).to(pytorch_device) for i in train_grid]
                train_vox_ten = [torch.from_numpy(i).to(pytorch_device) for i in train_grid]
                point_label_tensor=train_vox_label.type(torch.LongTensor).to(pytorch_device)
        
                # forward + backward + optimize
                outputs = my_model(train_pt_fea_ten,train_grid_ten)
                loss = lovasz_softmax(torch.nn.functional.softmax(outputs), point_label_tensor,ignore=255) + loss_fun(outputs,point_label_tensor)
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())
            except Exception as error: 
                if exce_counter == 0:
                    print(error)
                exce_counter += 1
            
            # zero the parameter gradients
            optimizer.zero_grad() #将梯度初始化为0
            pbar.update(1) #每次更新进度条长度
            start_training=True
            global_iter += 1 #一个epoch中的步数
        pbar.close() #进度条关闭
        epoch += 1  #下一个epoch训练

if __name__ == '__main__':
    # Training settings
    #建立解析对象
    parser = argparse.ArgumentParser(description='')
    #增加属性
    parser.add_argument('-d', '--data_dir', default='data')
    parser.add_argument('-p', '--model_save_path', default='./SemKITTI_PolarSeg.pt')
    parser.add_argument('-m', '--model', choices=['polar','traditional'], default='polar', help='training model: polar or traditional (default: polar)')
    parser.add_argument('-s', '--grid_size', nargs='+', type=int, default = [480,360,32], help='grid size of BEV representation (default: [480,360,32])')
    parser.add_argument('--train_batch_size', type=int, default=2, help='batch size for training (default: 2)')
    parser.add_argument('--val_batch_size', type=int, default=2, help='batch size for validation (default: 2)')
    parser.add_argument('--check_iter', type=int, default=4000, help='validation interval (default: 4000)')
    #将属性给予args实例
    args = parser.parse_args()
    if not len(args.grid_size) == 3:
        raise Exception('Invalid grid size! Grid size should have 3 dimensions.')
    #打印出命令行参数
    print(' '.join(sys.argv))
    print(args)
    main(args)
