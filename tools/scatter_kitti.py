import os
import sys
# 获取项目根目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# 添加项目根目录到 sys.path
sys.path.append(project_root)
os.environ["CUDA_VISIBLE_DEVICES"] = '5,'
import yaml
import torch
import datetime
import matplotlib.pyplot as plt
import numpy as np
from dataset.kitti_dataset import SemanticKITTI, point_image_dataset_semkitti, collate_fn_default
from easydict import EasyDict
from argparse import ArgumentParser

from tqdm import tqdm


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
    parser.add_argument('--gpu', type=int, nargs='+', default=(5,), help='specify gpu devices')
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--show_dataset",default=True, action='store_true')
    parser.add_argument('--config_path', default='../config/semantickitti.yaml')
    parser.add_argument('--root',type=str,default = '/data/xzy/elon/seg',help='the root directory to save scatter-kitti')
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
    # 1:  "car" 2: "bicycle" 3: "motorcycle" 4: "truck" 5: "other-vehicle" 6: "person" 7: "bicyclist" 8: "motorcyclist" 9: "road" # 10:"parking" 
    # 11:"sidewalk" 12:"other-ground" 13:"building" 14: "fence" 15: "vegetation" 16: "trunk" 17: "terrain"18: "pole" 19: "traffic-sign"      
    
    
    if specific_label in [1,4,18,2,8,14]:
        point_numb = 1
    elif specific_label in [3,19]:
        point_numb = 1
    elif specific_label in [5,12,6,7]:
        point_numb = 1
    elif specific_label in [13,10]:
        point_numb = 2
    else: # 9 11 16 15 17
        point_numb = 2
    
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
      
    pt_dataset = SemanticKITTI(config, data_path=data_config['data_path'], imageset='val', num_vote=data_config["batch_size"])
    
    dataset_loader = torch.utils.data.DataLoader(
                dataset=point_image_dataset_semkitti(pt_dataset, config),
                batch_size=data_config["batch_size"],
                collate_fn=collate_fn_default,
                shuffle=data_config["shuffle"],
                num_workers=data_config["num_workers"]
            )    
    
    pos2neg = config['target_label']
    total_counts = np.zeros(20)
    total_gt_counts = np.zeros(20)
    
    for  batch_idx, batch in enumerate(tqdm(dataset_loader)):
        list_masks = []
        
        img_indices = batch['img_indices']
        p2img_idx = batch['point2img_index']
        points = batch['points']
        label = batch['proj_label'].squeeze(dim=0).permute(2, 1, 0).numpy()
        gt_label = batch['labels']
        instance_label = batch['proj_instance_label'].squeeze(dim=0).permute(2, 1, 0).numpy()
        
        image = batch['img'].squeeze(dim=0).permute(1, 2, 0)*255
        
        # 获得图像中的label
        unique_labels = np.unique(label[label > 0])

       
        H, W, C = image.shape
        scatter_kitti = np.zeros((W, H)) # 初始化scatter-kitti
        for i in unique_labels:   
            pos_point, pos_label = get_pixel_coordinates(label,instance_label, i)
        
            if not np.any(pos_point):
                continue  
            
            scatter_kitti[pos_point[:,0],pos_point[:,1]] = i
            
        
        path = batch['path'][0]
        root = config.root
        basename = os.path.basename(path).split('.')[0]  
        
        if config.show_dataset:
            img_filename = os.path.join(root,  basename + ".png")         
            colorMap = np.array(config.colorMap, dtype=np.int32)
            show_scatter_dataset(scatter_kitti.astype(int), image, colorMap, img_filename)
        
        scatter_kitti_name = os.path.join(root,  basename + ".npy") 
        np.save(scatter_kitti_name, scatter_kitti)
        
        
   
        
      