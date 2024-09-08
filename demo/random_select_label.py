import os
import yaml
import torch
import matplotlib.pyplot as plt
import numpy as np
from dataset.kitti_dataset import SemanticKITTI, point_image_dataset_semkitti, collate_fn_default
from easydict import EasyDict
from argparse import ArgumentParser
from visualize.point2cam import point2cam_label
from tqdm import tqdm
from visualize.point2ply import labeled_point2ply
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
    parser.add_argument('--gpu', type=int, nargs='+', default=(1,), help='specify gpu devices')
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument('--root',type=str,default = '/data/elon',help='the root directory to save images')
    parser.add_argument('--config_path', default='/home/elon/Workshops/Scatter-annotation/config/semantickitti.yaml')
    
    
 
    # debug
    parser.add_argument('--debug', default=False, action='store_true')

    args = parser.parse_args()
    config = load_yaml(args.config_path)
    config.update(vars(args))  # override the configuration using the value in args


    return EasyDict(config)



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
    
    for  batch_idx, batch in enumerate(tqdm(dataset_loader)):
        label = batch['labels'].numpy()
        proj_label = batch['proj_label'].squeeze(dim=0)
        # proj_label = np.transpose(proj_label, (1, 2, 0))

        image = batch['img'].squeeze(dim=0).permute(1, 2, 0)*255
        points = batch['points']   
        # ---------------- visualize point to camera ------------------ #
        with torch.no_grad():
            path = batch['path'][0]
            root = '/data/elon/'
            basename = os.path.basename(path).split('.')[0]  
            img_filename = os.path.join(root,  basename + ".png") 
            colorMap = np.array(config.colorMap, dtype=np.int32)
            # point2cam(data_dict['proj_xyzi'][idx].detach().cpu(), data_dict['img'][idx].detach().cpu(), img_filename)           
            # point2cam_label(proj_label, image.permute(2, 0, 1), img_filename,colorMap)
            
            point_filename = os.path.join(root, basename + ".ply")
            non_zero_indices = np.where(label != 0)[0]  # 返回非零索引的数组
            # 计算需要置零的数量
            num_to_zero = int(0.01 * len(non_zero_indices))  # 95%的非零数目
            # 随机选择要置零的索引
            indices_to_zero = np.random.choice(non_zero_indices, size=num_to_zero, replace=False)  # 无放回抽样
            # label[indices_to_zero] = 0  # 设置为零
            labeled_point2ply(points[non_zero_indices], label[non_zero_indices],point_filename,colorMap)                
      