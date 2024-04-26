import os
import yaml
import torch
import matplotlib.pyplot as plt
import numpy as np
from dataset.nuscenes_dataset import SemanticKITTI, point_image_dataset_semkitti, nuScenes, point_image_dataset_nus, collate_fn_default
from easydict import EasyDict
from argparse import ArgumentParser
from visualize.point2cam import point2cam_label
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
    parser.add_argument('--gpu', type=int, nargs='+', default=(1,), help='specify gpu devices')
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument('--root',type=str,default = '/data/elon',help='the root directory to save images')
    parser.add_argument('--config_path', default='/home/elon/Projects/segment-anything-main/create_image_label/nuscenes.yaml')
    parser.add_argument('--sam_checkpoint', type=str, default="/home/elon/Projects/segment-anything-main/checkpoint/sam_vit_h_4b8939.pth",
                        help='Path to the SAM checkpoint file')
    parser.add_argument('--model_type', type=str, default="vit_h", help='Type of the model (e.g., vit_h)')
    parser.add_argument('--device', type=str, default="cuda", help='Device to run the model on (e.g., cuda)')
    parser.add_argument('--get_label', default=11, type=int, help='1: "car", 2: "bicycle", 3: "motorcycle", 4: "truck", '
                                                                '5: "other-vehicle", 6: "person", 7: "bicyclist", '
                                                                '8: "motorcyclist", 9: "road", 10: "parking", '
                                                                '11: "sidewalk", 12: "other-ground", 13: "building", '
                                                                '14: "fence", 15: "vegetation", 16: "trunk", '
                                                                '17: "terrain", 18: "pole", 19: "traffic-sign"')
 
    # debug
    parser.add_argument('--debug', default=False, action='store_true')

    args = parser.parse_args()
    config = load_yaml(args.config_path)
    config.update(vars(args))  # override the configuration using the value in args


    return EasyDict(config)


def get_neg_coordinates(all_labels, target_label):
    
    assert all_labels.shape[0] == 1, "第一个维度必须等于1" # label的形状[1, 1226,370]
    # 找到标签不等于255的像素位置
    sel_coordinates = []
    for index, item in enumerate(target_label):
        
        indices = np.where(all_labels[0] == target_label[index])
    
        if not np.any(indices):
            continue
            
        # 获取像素位置坐标
        coordinates = np.column_stack((indices[0], indices[1]))
    
        if coordinates.shape[0] > 10:
            selected_indices = np.random.choice(coordinates.shape[0], size= 1, replace=False)
       
            sel_coordinates.extend(coordinates[selected_indices])
        else:
            sel_coordinates.extend(coordinates)
   
    N = len(sel_coordinates)
    
    neg_label = np.zeros(N)
    return np.array(sel_coordinates), neg_label

def get_pixel_coordinates(all_labels, instance_label, specific_label):
    assert all_labels.shape[0] == 1, "第一个维度必须等于1" # label的形状[1, 1226,370]
    
    # 找到标签等于1且实例标签不重复的像素位置
    unique_instances = np.unique(instance_label[0][all_labels[0] == specific_label])
    coordinates = []
    selected_labels = []
    
    '''
    1:  "car" 2: "bicycle" 3: "motorcycle" 4: "truck" 5: "other-vehicle" 6: "person" 
    7: "bicyclist" 8: "motorcyclist" 9: "road" # 10:"parking" 11:"sidewalk" 
    12:"other-ground" 13:"building" 14: "fence" 15: "vegetation" 16: "trunk" 
    17: "terrain"18: "pole" 19: "traffic-sign"
    
    '''
      
    if specific_label in [3,7,8]:
        point_numb = 1
    elif specific_label in [14,1,4,2]:
        point_numb = 2
    elif specific_label in [5,12,6]:
        point_numb = 3
    elif specific_label in [13,10]:
        point_numb = 4
    else: # 9 11 16 15 17
        point_numb = 5
    
    for instance in unique_instances:
        # 找到当前实例对应的像素位置
        indices = np.where((all_labels[0] == specific_label) & (instance_label[0] == instance))
        instance_coordinates = np.column_stack((indices[0], indices[1]))
        instance_label_values = all_labels[0][indices[0], indices[1]]
        
        # 随机选择point_numb个标签
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



def show_mask(image, mask, path, color=[153, 204, 255]):
    visualized_image = np.copy(image)
    
    visualized_image[mask.squeeze()] = color  # 淡蓝色
    visualized_image = visualized_image.astype(np.uint8) 
    
    plt.imsave(path, visualized_image)

def show_points(coords_pos,coords_neg, image, img_filename):
    """Notice

   在 Matplotlib 中, imshow 函数默认情况下会将数组的第一个维度作为 y 轴，第二个维度作为 x 轴。
   因此，当你使用 scatter 函数时，如果你的坐标是 (y, x) 格式，你需要将其传递给 scatter 函数的参数 x 和 y。
   这与我们通常在数组中使用的顺序相反，因为通常我们使用 (row, column) 的格式。
    """
    plt.imshow(image.numpy().astype(np.uint8))
    plt.axis('off')
    plt.scatter(coords_pos[:, 0], coords_pos[:, 1], color='red', marker='.',s=50)  # 注意：x轴和y轴与数组形状相反
    plt.scatter(coords_neg[:, 0], coords_neg[:, 1], color='blue', marker='^',s=16)  # 注意：x轴和y轴与数组形状相反
    plt.savefig(img_filename, bbox_inches='tight', pad_inches=0)
    plt.clf()  # 清除图形状态


if __name__ == '__main__':
    # parameters
    config = parse_config()
    print(config)

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, config.gpu))
    num_gpu = len(config.gpu)
 
    # 加载SAM模型
    sam_checkpoint = config.sam_checkpoint 
    model_type = config.model_type
    device = config.device

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)    
    
    
    # 读取数据
    val_config = config['dataset_params']['val_data_loader']
    
    get_label = config.get_label
    
    pt_dataset = nuScenes(config, data_path=val_config['data_path'], imageset='val', num_vote=val_config["batch_size"])
    
    dataset_loader = torch.utils.data.DataLoader(
                dataset=point_image_dataset_nus(pt_dataset, config, val_config, num_vote=val_config["batch_size"]),
                batch_size=val_config["batch_size"],
                collate_fn=collate_fn_default,
                shuffle=val_config["shuffle"],
                num_workers=val_config["num_workers"]
            )    
    
    pos2neg = config['target_label']
    
    for  batch_idx, batch in enumerate(tqdm(dataset_loader)):
       
        label = batch['proj_label'].squeeze(dim=0).permute(2, 1, 0).numpy()
        instance_label = batch['proj_instance_label'].squeeze(dim=0).permute(2, 1, 0).numpy()
        image = batch['img'].squeeze(dim=0).permute(1, 2, 0)*255
          
        neg_point, neg_label = get_neg_coordinates(label,pos2neg[get_label])
        pos_point, pos_label = get_pixel_coordinates(label,instance_label, get_label)
        
        if not np.any(pos_point):
            raise ValueError("No elements found in pos_point")       
        if not np.any(neg_point):
            input_point = pos_point
            input_label = pos_label
        else:
            input_point = np.concatenate((pos_point, neg_point), axis=0)
            input_label = np.concatenate((pos_label, neg_label), axis=0)        
            
        predictor.set_image(image.numpy().astype(np.uint8))
        
        masks, scores, logits = predictor.predict(
                                point_coords=input_point,
                                point_labels=input_label,
                                multimask_output=True,
                                )
        
        mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask
        
        masks, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            mask_input=mask_input[None, :, :],
            multimask_output=False,
        )        
        path = batch['path'][0]
        root = config.root
        basename = os.path.basename(path).split('.')[0]  
        img_filename = os.path.join(root,  basename + ".jpg") 

        show_points(pos_point, neg_point, image, img_filename)
        

        
        # colorMap = np.array([[0, 0, 0],         # 0 'noise'
        #                     [255, 120, 50],     # 1 'barrier'
        #                     [100, 230, 245],    # 2  'bicycle'
        #                     [135, 60, 0],       # 3  'bus'
        #                     [100, 150, 245],    # 4  'car'
        #                     [100, 80, 250],     # 5  'construction_vehicle'
        #                     [30, 60, 150],      # 6  'motorcycle'
        #                     [255, 30, 30],      # 7  'pedestrian'
        #                     [255, 0, 0],        # 8   'traffic_cone'
        #                     [255, 240, 150],    # 9   'trailer'
        #                     [80, 30, 180],      # 10  'truck'
        #                     [255, 0, 255],      # 11  'driveable_surface'
        #                     [175, 0, 75],       # 12  'other_flat'
        #                     [75, 0, 75] ,       # 13  'sidewalk'
        #                     [150, 240, 80],     # 14  'terrain'
        #                     [255, 200, 0],      # 15 'manmade'
        #                     [0, 175, 0],        # 16   'vegetation'
                   
        #             ]).astype(np.int32)
        
        img_filename = os.path.join(root,  basename + ".png")
        colorMap = np.array(config.colorMap, dtype=np.int32)
        show_mask(image, masks, img_filename, color= colorMap[get_label][::-1])
        

        
        # ---------------- visualize point to camera ------------------ #
        # with torch.no_grad():
        #     path = batch['path'][0]
        #     root = '/data/elon'
        #     basename = os.path.basename(path).split('.')[0]  
        #     img_filename = os.path.join(root,  basename + ".png") 
        #     for idx in range(batch['batch_size']):
        #         # point2cam(data_dict['proj_xyzi'][idx].detach().cpu(), data_dict['img'][idx].detach().cpu(), img_filename)           
        #         point2cam_label(batch['proj_label'][idx].detach().cpu(), batch['img'][idx].detach().cpu(), img_filename)
                
                
        
      