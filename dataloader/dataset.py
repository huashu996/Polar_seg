#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SemKITTI dataloader
"""
import os
import numpy as np
import torch
import random
import time
import numba as nb
import yaml
from torch.utils import data

#KITTI数据
class SemKITTI(data.Dataset):
    #数据初始化加载数据
    def __init__(self, data_path, imageset = 'train', return_ref = False):
        self.return_ref = return_ref
        with open("semantic-kitti.yaml", 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml['learning_map']
        self.imageset = imageset
        if imageset == 'train':
            split = semkittiyaml['split']['train']
        elif imageset == 'val':
            split = semkittiyaml['split']['valid']
        elif imageset == 'test':
            split = semkittiyaml['split']['test']
        else:
            raise Exception('Split must be train/val/test')
        
        self.im_idx = []
        for i_folder in split:
            self.im_idx += absoluteFilePaths('/'.join([data_path,str(i_folder).zfill(2),'velodyne']))
        self.im_idx.sort()
    #总共多少帧
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)
    
    def __getitem__(self, index):
        #velodyne数据格式转换
        #这个-1代表的意思就是，我不知道可以分成多少行，但是我的需要是分成4列，多少行我不关心
        raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))#fromfile将每个bin文件用float32读取，转成4列

        #lable数据转换
        if self.imageset == 'test':
        #zeros_like函数主要是想实现构造一个矩阵W_update，其维度与矩阵W一致，并为其初始化为全0
        #np.expand_dims(a,axis=)即在 a 的相应的axis轴上扩展维度  如果axis=1，那么改变这个形状的1就会插在第二的位置 一列变成了一个有厚度的数
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:,0],dtype=int),axis=1) #

        else:
            annotated_data = np.fromfile(self.im_idx[index].replace('velodyne','labels')[:-3]+'label', dtype=np.int32).reshape((-1,1))
            annotated_data = annotated_data & 0xFFFF #删除高的16
            #vectorize向量化数组
            annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)#

        #uint8是专门储存图像的数据格式0～255
        data_tuple = (raw_data[:,:3], annotated_data.astype(np.uint8))#返回转换后的velodyne和lable  [:,:3]前三列0,1,2,3  一列lable
        if self.return_ref:
            data_tuple += (raw_data[:,3],)
        return data_tuple #返回数据数组n*4矩阵  
#获得文件路径
def absoluteFilePaths(directory):
   for dirpath,_,filenames in os.walk(directory):
       for f in filenames:
           yield os.path.abspath(os.path.join(dirpath, f))

#VOXEL数据
class voxel_dataset(data.Dataset):
  def __init__(self, in_dataset, grid_size, rotate_aug = False, flip_aug = False, ignore_label = 255, return_test = False,
            fixed_volume_space= False, max_volume_space = [50,50,1.5], min_volume_space = [-50,-50,-3]):
        'Initialization'
        self.point_cloud_dataset = in_dataset
        self.grid_size = np.asarray(grid_size)
        self.rotate_aug = rotate_aug
        self.ignore_label = ignore_label
        self.return_test = return_test
        self.flip_aug = flip_aug
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.point_cloud_dataset)

  def __getitem__(self, index):
        'Generates one sample of data'
        data = self.point_cloud_dataset[index]
        if len(data) == 2:
            xyz,labels = data
        elif len(data) == 3:
            xyz,labels,sig = data
            if len(sig.shape) == 2: sig = np.squeeze(sig)
        else: raise Exception('Return invalid data tuple')
        
        # random data augmentation by rotation
        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random()*360)
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:,:2] = np.dot( xyz[:,:2],j)

        # random data augmentation by flip x , y or x+y
        if self.flip_aug:
            flip_type = np.random.choice(4,1)
            if flip_type==1:
                xyz[:,0] = -xyz[:,0]
            elif flip_type==2:
                xyz[:,1] = -xyz[:,1]
            elif flip_type==3:
                xyz[:,:2] = -xyz[:,:2]

        max_bound = np.percentile(xyz,100,axis = 0)
        min_bound = np.percentile(xyz,0,axis = 0)
        
        if self.fixed_volume_space:
            max_bound = np.asarray(self.max_volume_space)
            min_bound = np.asarray(self.min_volume_space)

        # get grid index
        crop_range = max_bound - min_bound
        cur_grid_size = self.grid_size
        
        intervals = crop_range/(cur_grid_size-1)
        if (intervals==0).any(): print("Zero interval!")
        
        grid_ind = (np.floor((np.clip(xyz,min_bound,max_bound)-min_bound)/intervals)).astype(np.int)

        # process voxel position
        voxel_position = np.zeros(self.grid_size,dtype = np.float32)
        dim_array = np.ones(len(self.grid_size)+1,int)
        dim_array[0] = -1 
        voxel_position = np.indices(self.grid_size)*intervals.reshape(dim_array) + min_bound.reshape(dim_array)

        # process labels
        processed_label = np.ones(self.grid_size,dtype = np.uint8)*self.ignore_label
        label_voxel_pair = np.concatenate([grid_ind,labels],axis = 1)
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:,0],grid_ind[:,1],grid_ind[:,2])),:]
        processed_label = nb_process_label(np.copy(processed_label),label_voxel_pair)
        
        data_tuple = (voxel_position,processed_label)

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5)*intervals + min_bound
        return_xyz = xyz - voxel_centers
        return_xyz = np.concatenate((return_xyz,xyz),axis = 1)
        
        if len(data) == 2:
            return_fea = return_xyz
        elif len(data) == 3:
            return_fea = np.concatenate((return_xyz,sig[...,np.newaxis]),axis = 1)
        
        if self.return_test:
            data_tuple += (grid_ind,labels,return_fea,index)
        else:
            data_tuple += (grid_ind,labels,return_fea)
        return data_tuple

# transformation between Cartesian coordinates and polar coordinates
#xyz转化为极坐标
def cart2polar(input_xyz):  
    rho = np.sqrt(input_xyz[:,0]**2 + input_xyz[:,1]**2)
    phi = np.arctan2(input_xyz[:,1],input_xyz[:,0])
    return np.stack((rho,phi,input_xyz[:,2]),axis=1) #n*3
#极坐标转化为XYZ   
def polar2cat(input_xyz_polar):
    x = input_xyz_polar[0]*np.cos(input_xyz_polar[1])
    y = input_xyz_polar[0]*np.sin(input_xyz_polar[1])
    return np.stack((x,y,input_xyz_polar[2]),axis=0) 


#球状数据
class spherical_dataset(data.Dataset):
  def __init__(self, in_dataset, grid_size, rotate_aug = False, flip_aug = False, ignore_label = 255, return_test = False,
               fixed_volume_space= False, max_volume_space = [50,np.pi,1.5], min_volume_space = [3,-np.pi,-3]):
        'Initialization'
        self.point_cloud_dataset = in_dataset
        print(self.point_cloud_dataset)
        self.grid_size = np.asarray(grid_size) #480,360,32
        self.rotate_aug = rotate_aug
        self.flip_aug = flip_aug
        self.ignore_label = ignore_label
        self.return_test = return_test
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.point_cloud_dataset)

  def __getitem__(self, index):
        'Generates one sample of data'
        data = self.point_cloud_dataset[index]
        if len(data) == 2:
            xyz,labels = data  #xyz n*3
        elif len(data) == 3:
            xyz,labels,sig = data
            if len(sig.shape) == 2: sig = np.squeeze(sig) #函数将表示向量的数组转换为秩为1的数组
        else: raise Exception('Return invalid data tuple')
        
        # random data augmentation by rotation
        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random()*360)#弧度与角度的转换
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:,:2] = np.dot( xyz[:,:2],j)#矩阵点乘法

        # random data augmentation by flip x , y or x+y
        if self.flip_aug:
            flip_type = np.random.choice(4,1) #从中随意选取一个数
            if flip_type==1:
                xyz[:,0] = -xyz[:,0]
            elif flip_type==2:
                xyz[:,1] = -xyz[:,1]
            elif flip_type==3:
                xyz[:,:2] = -xyz[:,:2]

        # convert coordinate into polar coordinates
        xyz_pol = cart2polar(xyz)
        '''
        np.percentile(a, 50) #50%的分位数，就是a里排序之后的中位数3.5
        np.percentile(a, 50, axis=0) #axis为0，在纵列上求
        array([[ 6.5,  4.5,  2.5]])
        np.percentile(a, 50, axis=1) #axis为1，在横行上求
        array([ 7.,  2.]
        '''
        #xyz_pol n*3
        max_bound_r = np.percentile(xyz_pol[:,0],100,axis = 0) #最大半径 一个数
        min_bound_r = np.percentile(xyz_pol[:,0],0,axis = 0)  #最小半径 一个数
        max_bound = np.max(xyz_pol[:,1:],axis = 0) #最大范围  每一列最大值  两个数
        min_bound = np.min(xyz_pol[:,1:],axis = 0) #最小范围  每一列最小值  两个数
        max_bound = np.concatenate(([max_bound_r],max_bound))#concatenate拼接
        min_bound = np.concatenate(([min_bound_r],min_bound))
        if self.fixed_volume_space:
            max_bound = np.asarray(self.max_volume_space)
            min_bound = np.asarray(self.min_volume_space)

        # get grid index
        crop_range = max_bound - min_bound  #最远最高  最近最低
        cur_grid_size = self.grid_size  #480,360,32
        intervals = crop_range/(cur_grid_size-1) #半径480份  角度360份 高度32份 单位圆环格

        if (intervals==0).any(): print("Zero interval!")
        #np.clip是一个截取函数，用于截取数组中小于或者大于某值的部分，并使得被截取部分等于固定值。
        #np.floor()返回不大于输入参数的最大整数
        grid_ind = (np.floor((np.clip(xyz_pol,min_bound,max_bound)-min_bound)/intervals)).astype(np.int) #将点云划分到每个单元格内
        # process voxel position
        voxel_position = np.zeros(self.grid_size,dtype = np.float32)# [0 0 0]
        dim_array = np.ones(len(self.grid_size)+1,int)# 4
        dim_array[0] = -1 #[-1 1 1 1]
        voxel_position = np.indices(self.grid_size)*intervals.reshape(dim_array) + min_bound.reshape(dim_array)
        # voxel_position = polar2cat(voxel_position)
        
        # process labels
        processed_label = np.ones(self.grid_size,dtype = np.uint8)*self.ignore_label
        label_voxel_pair = np.concatenate([grid_ind,labels],axis = 1)
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:,0],grid_ind[:,1],grid_ind[:,2])),:]#按第一层半径 第二层角度 第三层高度进行排序
        processed_label = nb_process_label(np.copy(processed_label),label_voxel_pair)
        # data_tuple = (voxel_position,processed_label)

        # prepare visiblity feature
        # find max distance index in each angle,height pair
        valid_label = np.zeros_like(processed_label,dtype=bool)
        valid_label[grid_ind[:,0],grid_ind[:,1],grid_ind[:,2]] = True
        valid_label = valid_label[::-1]
        max_distance_index = np.argmax(valid_label,axis=0)
        max_distance = max_bound[0]-intervals[0]*(max_distance_index)
        distance_feature = np.expand_dims(max_distance, axis=2)-np.transpose(voxel_position[0],(1,2,0))
        distance_feature = np.transpose(distance_feature,(1,2,0))
        # convert to boolean feature
        distance_feature = (distance_feature>0)*-1.
        distance_feature[grid_ind[:,2],grid_ind[:,0],grid_ind[:,1]]=1.

        data_tuple = (distance_feature,processed_label)

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5)*intervals + min_bound
        return_xyz = xyz_pol - voxel_centers
        return_xyz = np.concatenate((return_xyz,xyz_pol,xyz[:,:2]),axis = 1)

        if len(data) == 2:
            return_fea = return_xyz
        elif len(data) == 3:
            return_fea = np.concatenate((return_xyz,sig[...,np.newaxis]),axis = 1)
        
        if self.return_test:
            data_tuple += (grid_ind,labels,return_fea,index)
        else:
            data_tuple += (grid_ind,labels,return_fea)
        return data_tuple
    
@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])',nopython=True,cache=True,parallel = False)
def nb_process_label(processed_label,sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,),dtype = np.uint16)
    counter[sorted_label_voxel_pair[0,3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0,:3]
    for i in range(1,sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i,:3]
        if not np.all(np.equal(cur_ind,cur_sear_ind)):
            processed_label[cur_sear_ind[0],cur_sear_ind[1],cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,),dtype = np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i,3]] += 1
    processed_label[cur_sear_ind[0],cur_sear_ind[1],cur_sear_ind[2]] = np.argmax(counter)
    return processed_label

def collate_fn_BEV(data):
    data2stack=np.stack([d[0] for d in data]).astype(np.float32)
    label2stack=np.stack([d[1] for d in data])
    grid_ind_stack = [d[2] for d in data]
    point_label = [d[3] for d in data]
    xyz = [d[4] for d in data]
    return torch.from_numpy(data2stack),torch.from_numpy(label2stack),grid_ind_stack,point_label,xyz

def collate_fn_BEV_test(data):    
    data2stack=np.stack([d[0] for d in data]).astype(np.float32)
    label2stack=np.stack([d[1] for d in data])
    grid_ind_stack = [d[2] for d in data]
    point_label = [d[3] for d in data]
    xyz = [d[4] for d in data]
    index = [d[5] for d in data]
    return torch.from_numpy(data2stack),torch.from_numpy(label2stack),grid_ind_stack,point_label,xyz,index

# load Semantic KITTI class info
with open("semantic-kitti.yaml", 'r') as stream:
    semkittiyaml = yaml.safe_load(stream)
SemKITTI_label_name = dict()
for i in sorted(list(semkittiyaml['learning_map'].keys()))[::-1]:
    SemKITTI_label_name[semkittiyaml['learning_map'][i]] = semkittiyaml['labels'][i]
