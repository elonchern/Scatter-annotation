import os
import sys
# 获取项目根目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# 添加项目根目录到 sys.path
sys.path.append(project_root)
import yaml
import torch
import datetime
import matplotlib.pyplot as plt
import numpy as np
from dataset.nuscenes_dataset import nuScenes, point_image_dataset_nus,collate_fn_default
from easydict import EasyDict
from argparse import ArgumentParser

from tqdm import tqdm
from segment_anything import sam_model_registry, SamPredictor


def load_yaml(file_name):
    with open(file_name, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            config = yaml.load(f)
    return config

def parse_config():
    parser = ArgumentParser()
    # general
    parser.add_argument('--gpu', type=int, nargs='+', default=(0,), help='specify gpu devices')
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--show_dataset",default=False, action='store_true')
    parser.add_argument('--config_path', default='../config/nuscenes.yaml')
    parser.add_argument('--root',type=str,default = '/data/elon',help='the root directory to save images') 
    # debug
    parser.add_argument('--debug', default=False, action='store_true')

    args = parser.parse_args()
    config = load_yaml(args.config_path)
    config.update(vars(args))  # override the configuration using the value in args


    return EasyDict(config)


def get_pixel_coordinates(all_labels, instance_label, specific_label):
    assert all_labels.shape[0] == 1, "第一个维度必须等于1" # label的形状[1, 1226,370]
    
    # 找到标签等于1且实例标签不重复的像素位置
    unique_instances = np.unique(instance_label[0][all_labels[0] == specific_label])
    coordinates = []
    selected_labels = []
    
    """
    0'noise' 1'barrier' 2'bicycle' 3'bus' 4'car' 5'construction_vehicle' 
    6'motorcycle' 7'pedestrian' 8'traffic_cone' 9'trailer' 10'truck' 11'driveable_surface' 
    12'other_flat' 13'sidewalk' 14'terrain' 15'manmade' 16'vegetation'
    """
      
    if specific_label in [1,2,3,5,9]:
        point_numb = 2
    elif specific_label in [6]:
        point_numb = 2
    elif specific_label in [12,10]:
        point_numb = 3
    elif specific_label in [7,8,11,12,13,14]:
        point_numb = 4
    else: #15,16
        point_numb = 4
    
    for instance in unique_instances:
        # 找到当前实例对应的像素位置
        indices = np.where((all_labels[0] == specific_label) & (instance_label[0] == instance))
        instance_coordinates = np.column_stack((indices[0], indices[1]))
        instance_label_values = all_labels[0][indices[0], indices[1]]
        
        # 随机选择两个标签
        if instance_coordinates.shape[0] > point_numb:
            selected_indices = np.random.choice(instance_coordinates.shape[0], size=point_numb, replace=False)
            coordinates.extend(instance_coordinates[selected_indices])
            selected_labels.extend(instance_label_values[selected_indices])
        else:
            coordinates.extend(instance_coordinates)
            selected_labels.extend(instance_label_values)
            
    N = len(selected_labels)
    pos_label = np.ones(N)
    return np.array(coordinates),  pos_label


def get_scatter_dataset(image, label, instance_label, pos2neg,num_classes):
    # list_masks = []
    
    # 获得图像中的label
    unique_labels = np.unique(label[label > 0])
    
    H, W, C = image.shape
    scatter_dataset = np.zeros((W, H))
    for i in unique_labels:   
    
        pos_point, pos_label = get_pixel_coordinates(label,instance_label, i)
    
        if not np.any(pos_point):
            continue  
        
        scatter_dataset[pos_point[:,0],pos_point[:,1]] = i      

    counts = np.bincount(scatter_dataset.flatten().astype(int), minlength=num_classes)
    

    return counts, scatter_dataset



def majority_vote(predictions):
    num_points = len(predictions[0])
    
    
    final_predictions = np.zeros(num_points, dtype=np.int32)
    
    for i in range(num_points):
        counts = np.bincount(predictions[:, i])
        # 将0类排除在外
        counts[0] = 0
        
        # 如果所有预测结果都是0，则将标签置为0
        if np.all(predictions[:, i] == 0):
            final_predictions[i] = 0
        else:
            # 找到投票最多的标签
            max_count = np.max(counts)
            max_indices = np.where(counts == max_count)[0]
            if len(max_indices) == 1:
                final_predictions[i] = max_indices[0]
            else:
                final_predictions[i] = 0
    
    return final_predictions


def show_scatter_dataset(mask, image, colors, img_filename):
    
    nonzero_indices = np.nonzero(mask)
    plt.imshow(image.numpy().astype(np.uint8))
    plt.axis('off')
    for i in range(len(nonzero_indices[0])):
        coords = (nonzero_indices[0][i], nonzero_indices[1][i])
        label = mask[coords]
        plt.scatter(int(coords[0]), int(coords[1]), color=(colors[label][0]/255, colors[label][1]/255, colors[label][2]/255), marker='.',s=50) 
        
    plt.savefig(img_filename, bbox_inches='tight', pad_inches=0)    
        
    plt.clf()  # 清除之前的图形


if __name__ == '__main__':
    # parameters
    config = parse_config()
    print(config)

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, config.gpu))
    num_gpu = len(config.gpu)
    
    
    # 读取数据
    data_config = config['dataset_params']['data_loader']
    
    pt_dataset = nuScenes(config, data_path=data_config['data_path'], imageset='val', num_vote=data_config["batch_size"])
    
    dataset_loader = torch.utils.data.DataLoader(
                dataset=point_image_dataset_nus(pt_dataset, config),
                batch_size=data_config["batch_size"],
                collate_fn=collate_fn_default,
                shuffle=data_config["shuffle"],
                num_workers=data_config["num_workers"]
            )        
    

    num_classes=config['dataset_params']['num_classes']
    pos2neg = config['target_label']
 

    total_gt_counts = np.zeros(17)
    for  batch_idx, batch in enumerate(tqdm(dataset_loader)):
        path = batch['path'][0]  
        
        points = batch['points']
        labels = batch['labels']
        
        label_0 = batch['proj_label_0'].squeeze(dim=0).permute(2, 1, 0).numpy()
        instance_label_0 = batch['proj_instance_label_0'].squeeze(dim=0).permute(2, 1, 0).numpy()
        image_0 = batch['img_0'].squeeze(dim=0).permute(1, 2, 0)*255
        img_indices_0 = batch['img_indices_0']
        p2img_idx_0 = batch['point2img_index_0']
        
        label_1 = batch['proj_label_1'].squeeze(dim=0).permute(2, 1, 0).numpy()
        instance_label_1 = batch['proj_instance_label_1'].squeeze(dim=0).permute(2, 1, 0).numpy()
        image_1 = batch['img_1'].squeeze(dim=0).permute(1, 2, 0)*255
        img_indices_1 = batch['img_indices_1']
        p2img_idx_1 = batch['point2img_index_1']
        
        label_2 = batch['proj_label_2'].squeeze(dim=0).permute(2, 1, 0).numpy()
        instance_label_2 = batch['proj_instance_label_2'].squeeze(dim=0).permute(2, 1, 0).numpy()
        image_2 = batch['img_2'].squeeze(dim=0).permute(1, 2, 0)*255
        img_indices_2 = batch['img_indices_2']
        p2img_idx_2 = batch['point2img_index_2']
        
        label_3 = batch['proj_label_3'].squeeze(dim=0).permute(2, 1, 0).numpy()
        instance_label_3 = batch['proj_instance_label_3'].squeeze(dim=0).permute(2, 1, 0).numpy()
        image_3 = batch['img_3'].squeeze(dim=0).permute(1, 2, 0)*255
        img_indices_3 = batch['img_indices_3']
        p2img_idx_3 = batch['point2img_index_3']
        
        label_4 = batch['proj_label_4'].squeeze(dim=0).permute(2, 1, 0).numpy()
        instance_label_4 = batch['proj_instance_label_4'].squeeze(dim=0).permute(2, 1, 0).numpy()
        image_4 = batch['img_4'].squeeze(dim=0).permute(1, 2, 0)*255
        img_indices_4 = batch['img_indices_4']
        p2img_idx_4 = batch['point2img_index_4']
        
        label_5 = batch['proj_label_5'].squeeze(dim=0).permute(2, 1, 0).numpy()
        instance_label_5 = batch['proj_instance_label_5'].squeeze(dim=0).permute(2, 1, 0).numpy()
        image_5 = batch['img_5'].squeeze(dim=0).permute(1, 2, 0)*255
        img_indices_5 = batch['img_indices_5']
        p2img_idx_5 = batch['point2img_index_5']
        
        counts_0, scatter_nuscenes_0 = get_scatter_dataset(image_0, label_0, instance_label_0, pos2neg, num_classes) # CMA_Front
        counts_1, scatter_nuscenes_1 = get_scatter_dataset(image_1, label_1, instance_label_1, pos2neg, num_classes) # CMA_Front_Left
        counts_2, scatter_nuscenes_2 = get_scatter_dataset(image_2, label_2, instance_label_2, pos2neg, num_classes) # CMA_Front_Right
        counts_3, scatter_nuscenes_3 = get_scatter_dataset(image_3, label_3, instance_label_3, pos2neg, num_classes) # CMA_Back
        counts_4, scatter_nuscenes_4 = get_scatter_dataset(image_4, label_4, instance_label_4, pos2neg, num_classes) # CMA_Back_Left
        counts_5, scatter_nuscenes_5 = get_scatter_dataset(image_5, label_5, instance_label_5, pos2neg, num_classes) # CMA_Back_Right
        
        
        path = batch['path'][0]
        root = config.root
        basename = os.path.basename(path).split('.')[0]  
        scatter_kitti_name = os.path.join(root,  basename + ".npz") 
        np.savez(scatter_kitti_name, 
                CMA_Front=scatter_nuscenes_0.astype(int), 
                CMA_Front_Left=scatter_nuscenes_1.astype(int), 
                CMA_Front_Right=scatter_nuscenes_2.astype(int), 
                CMA_Back=scatter_nuscenes_3.astype(int),
                CMA_Back_Left=scatter_nuscenes_4.astype(int),
                CMA_Back_Right=scatter_nuscenes_5.astype(int))
        
        
        if config.show_dataset:
            colorMap = np.array(config.colorMap, dtype=np.int32)
            img_filename = os.path.join(root,  basename + "_CMA_Front"+".png")  
            show_scatter_dataset(scatter_nuscenes_0.astype(int), image_0, colorMap, img_filename)
            
            img_filename = os.path.join(root,  basename + "_CMA_Front_Left"+".png")  
            show_scatter_dataset(scatter_nuscenes_1.astype(int), image_1, colorMap, img_filename)        
            
            img_filename = os.path.join(root,  basename + "_CMA_Front_Right"+".png")  
            show_scatter_dataset(scatter_nuscenes_2.astype(int), image_2, colorMap, img_filename)    
            
            img_filename = os.path.join(root,  basename + "_CMA_Back"+".png")  
            show_scatter_dataset(scatter_nuscenes_3.astype(int), image_3, colorMap, img_filename)     
            
            img_filename = os.path.join(root,  basename + "_CMA_Back_Left"+".png")  
            show_scatter_dataset(scatter_nuscenes_4.astype(int), image_4, colorMap, img_filename)    
            
            img_filename = os.path.join(root,  basename + "_CMA_Back_Right"+".png")  
            show_scatter_dataset(scatter_nuscenes_5.astype(int), image_5, colorMap, img_filename)           
        
        
    