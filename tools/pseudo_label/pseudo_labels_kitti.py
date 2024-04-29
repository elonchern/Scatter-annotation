import os
import sys
# 获取项目根目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
# 添加项目根目录到 sys.path
sys.path.append(project_root)
import yaml
import torch
import datetime
import matplotlib.pyplot as plt
import numpy as np
from dataset.kitti_dataset import SemanticKITTI, point_image_dataset_semkitti, collate_fn_default
from easydict import EasyDict
from argparse import ArgumentParser
from visualize.point2cam import point2cam_label,point2cam
from visualize.point2ply import labeled_point2ply
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
    parser.add_argument('--config_path', default='../../config/semantickitti.yaml')
    parser.add_argument('--sam_checkpoint', type=str, default="../../checkpoint/sam_vit_h_4b8939.pth",
                        help='Path to the SAM checkpoint file')
    parser.add_argument('--model_type', type=str, default="vit_h", help='Type of the model (e.g., vit_h)')
    parser.add_argument('--pseudo_type', type=str, default="image", help='Type of the pseudo (e.g., point, image)')
    parser.add_argument('--pseudo_labels_vis', type=bool, default=True)
    parser.add_argument('--device', type=str, default="cuda", help='Device to run the model on (e.g., cuda)')
    parser.add_argument('--root',type=str,default = '/data/elon/val_pred_spvcnn',help='the root directory to save pseudo labels') 
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
            selected_indices = np.random.choice(coordinates.shape[0], size= 10, replace=False)
       
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
    # 1:  "car" 2: "bicycle" 3: "motorcycle" 4: "truck" 5: "other-vehicle" 6: "person" 7: "bicyclist" 8: "motorcyclist" 9: "road" # 10:"parking" 
    # 11:"sidewalk" 12:"other-ground" 13:"building" 14: "fence" 15: "vegetation" 16: "trunk" 17: "terrain"18: "pole" 19: "traffic-sign"      
    
    
    if specific_label in [1,4,18,2,8,14]:
        point_numb = 5
    elif specific_label in [3,19]:
        point_numb = 2
    elif specific_label in [5,12,6,7]:
        point_numb = 10
    elif specific_label in [13,10]:
        point_numb = 15
    else: # 9 11 16 15 17
        point_numb = 20
    
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

def class_accuracy(labels, predictions, num_classes=20):
    class_accuracies = np.zeros(num_classes)
    for class_label in range(num_classes):
        class_indices = np.where(labels == class_label)[0]
        if len(class_indices) > 0:
            correct_predictions = np.sum(predictions[class_indices] == class_label)
            class_accuracies[class_label] = correct_predictions / len(class_indices)
    return class_accuracies

def compute_accuracy(true_labels, predicted_labels):
    
    # 确保两个标签数组的形状相同
    assert true_labels.shape == predicted_labels.shape, "标签数组形状不匹配"

    # 去除预测为0的标签
    nonzero_index = np.nonzero(predicted_labels)[0]
    
    # 根据索引获取相应的值
    true_labels = true_labels[nonzero_index]
    predicted_labels = predicted_labels[nonzero_index]

    # 计算预测正确的数量
    correct_count = np.sum(true_labels == predicted_labels)
    
    class_acc = class_accuracy(true_labels, predicted_labels, num_classes=20)

    # 计算准确性
    accuracy = correct_count / len(true_labels)
    
    return accuracy, class_acc


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
    sum_acc = 0
    num_classes=config['dataset_params']['num_classes']
    sum_class_acc = np.zeros(num_classes)
    count = 0
    for  batch_idx, batch in enumerate(tqdm(dataset_loader)):
        list_masks = []
       
        img_indices = batch['img_indices']
        p2img_idx = batch['point2img_index']
        points = batch['points']
        gt_label = batch['labels']
        label = batch['proj_label'].squeeze(dim=0).permute(2, 1, 0).numpy()
        instance_label = batch['proj_instance_label'].squeeze(dim=0).permute(2, 1, 0).numpy()
        
        image = batch['img'].squeeze(dim=0).permute(1, 2, 0)*255
        
        # 获得图像中的label
        unique_labels = np.unique(label[label > 0])

        # for 循环
        predictor.set_image(image.numpy().astype(np.uint8))
        
        for i in unique_labels:   
      
            neg_point, neg_label = get_neg_coordinates(label, pos2neg[i])
            pos_point, pos_label = get_pixel_coordinates(label,instance_label, i)
        
            if not np.any(pos_point):
                continue        
            if not np.any(neg_point):
                input_point = pos_point
                input_label = pos_label
            else:
                input_point = np.concatenate((pos_point, neg_point), axis=0)
                input_label = np.concatenate((pos_label, neg_label), axis=0)
                
            
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
            
            masks = masks.astype(int)
            masks = masks * i
             
            list_masks.append(masks)
            del masks,scores,logits
        
        merged_masks = np.zeros_like(list_masks[0])
        value_order = [18,19,17,16,15,12,5,13,6,11,9,14,4,1,10,8,7,3,2] 
        """
        1:  "car" 2: "bicycle" 3: "motorcycle" 4: "truck" 5: "other-vehicle" 6: "person" 7: "bicyclist" 
        8: "motorcyclist" 9: "road" # 10:"parking" 11:"sidewalk" 12:"other-ground" 13:"building"
        14: "fence" 15: "vegetation" 16: "trunk" 17: "terrain"18: "pole" 19: "traffic-sign"  
        """  
        for array_index, value in enumerate(value_order):
            for array in list_masks:
                max_value = np.max(array)
                if max_value == value:
                    indices = np.where(array == value)
                    merged_masks[0, indices[1], indices[2]] = value        
        
        
                             
        path = batch['path'][0]
        root = config.root # '/data/elon/segment/05/'
        basename = os.path.basename(path).split('.')[0] 
        if config.pseudo_type == "image":
            pseudo_img_filename = os.path.join(root,  basename + ".npy") 
            np.save(pseudo_img_filename, merged_masks.astype(np.uint8))
            if config.pseudo_labels_vis:
                img_filename = os.path.join(root,  basename + ".jpg") 
                merged_masks_transposed = np.transpose(merged_masks, (1, 2, 0)) # [1, 370, 1226] -> [370, 1226, 1]
                colorMap = np.array(config.colorMap, dtype=np.int32)
                point2cam_label(merged_masks_transposed, image.cpu().permute(2, 0, 1), img_filename,colorMap)  
            
             
        if config.pseudo_type == "point":
            merged_masks_transposed = np.transpose(merged_masks, (1, 2, 0)) # [1, 370, 1226] -> [370, 1226, 1]   
            proj_labels = merged_masks_transposed[img_indices[0][:, 0], img_indices[0][:, 1]]
            pseudo_labels = np.zeros(points.shape[0], dtype=np.int32)
            pseudo_labels[p2img_idx[0]] = proj_labels[:,0]
            pseudo_point_filename = os.path.join(root, basename + ".npy") 
            np.save(pseudo_point_filename, pseudo_labels.astype(np.uint8))
            if config.pseudo_labels_vis:
                point_filename = os.path.join(root, basename + ".ply")
                colorMap = np.array(config.colorMap, dtype=np.int32)
                labeled_point2ply(points, pseudo_labels,point_filename,colorMap) 
         
        
            # 统计pseudo_labels准确性
            accuracy, class_acc = compute_accuracy(gt_label.numpy(), pseudo_labels)
            count += 1
            sum_acc += accuracy
            sum_class_acc += class_acc

    if config.pseudo_type == "point":
        sum_result = sum_acc/count
        sum_class_result = sum_class_acc/count
            
        print(sum_class_result)          
        print(sum_result)    
    
     
        
      