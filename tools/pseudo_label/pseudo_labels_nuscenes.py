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
from dataset.nuscenes_dataset import  nuScenes, point_image_dataset_nus,collate_fn_default
from easydict import EasyDict
from argparse import ArgumentParser
from visualize.point2cam import point2cam_label, point2cam
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
    parser.add_argument('--gpu', type=int, nargs='+', default=(3,), help='specify gpu devices')
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument('--config_path', default='/userHome/xzy/Projects/elon/Scatter-annotation/config/nuscenes.yaml')
    parser.add_argument('--sam_checkpoint', type=str, default="/userHome/xzy/Projects/elon/Scatter-annotation/checkpoint/sam_vit_h_4b8939.pth",
                        help='Path to the SAM checkpoint file')
    parser.add_argument('--model_type', type=str, default="vit_h", help='Type of the model (e.g., vit_h)')
    parser.add_argument('--pseudo_type', type=str, default="image", help='Type of the pseudo (e.g., point, image)')
    parser.add_argument('--pseudo_labels_vis', type=bool, default=False)
    parser.add_argument('--device', type=str, default="cuda", help='Device to run the model on (e.g., cuda)')
    parser.add_argument('--root',type=str,default = '/data/xzy/elon/nuscenes/imageseg/v1.0-trainval/',help='the root directory to save pseudo labels')  
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
    
    """
    0'noise' 1'barrier' 2'bicycle' 3'bus' 4'car' 5'construction_vehicle' 
    6'motorcycle' 7'pedestrian' 8'traffic_cone' 9'trailer' 10'truck' 11'driveable_surface' 
    12'other_flat' 13'sidewalk' 14'terrain' 15'manmade' 16'vegetation'
    """
      
    if specific_label in [1,2,3,5,9]:
        point_numb = 5
    elif specific_label in [6]:
        point_numb = 2
    elif specific_label in [12,10]:
        point_numb = 10
    elif specific_label in [7,8,11,12,13,14]:
        point_numb = 15
    else: #15,16
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



def get_img_label(predictor, image, label, instance_label, pos2neg,config):
    list_masks = []
    
    # 获得图像中的label
    unique_labels = np.unique(label[label > 0])
    
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
   
    """
    0'noise' 1'barrier' 2'bicycle' 3'bus' 4'car' 5'construction_vehicle' 
    6'motorcycle' 7'pedestrian' 8'traffic_cone' 9'trailer' 10'truck' 11'driveable_surface' 
    12'other_flat' 13'sidewalk' 14'terrain' 15'manmade' 16'vegetation'
    """
    
    value_order = [8, 1, 2, 9, 10, 8, 12, 6, 7, 14, 16, 5, 3,4,15,13,11]    
    for array_index, value in enumerate(value_order):
        for array in list_masks:
            max_value = np.max(array)
            if max_value == value:
                
                indices = np.where(array == value)
                merged_masks[0, indices[1], indices[2]] = value        
    
    merged_masks_transposed = np.transpose(merged_masks, (1, 2, 0)) # [1, 370, 1226] -> [370, 1226, 1]
    if config.pseudo_labels_vis:
        colorMap = np.array(config.colorMap, dtype=np.int32)
        point2cam_label(merged_masks_transposed, image.cpu().permute(2, 0, 1), img_filename,colorMap)  
    
    return merged_masks_transposed, merged_masks


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

def class_accuracy(labels, predictions, num_classes=17):
    class_accuracies = np.zeros(num_classes)
    for class_label in range(num_classes):
        class_indices = np.where(labels == class_label)[0]
        if len(class_indices) > 0:
            correct_predictions = np.sum(predictions[class_indices] == class_label)
            class_accuracies[class_label] = correct_predictions / len(class_indices)
    return class_accuracies


def compute_accuracy(true_labels, predicted_labels,num_classes):
    
    # 确保两个标签数组的形状相同
    assert true_labels.shape == predicted_labels.shape, "标签数组形状不匹配"

    # 去除预测为0的标签
    nonzero_index = np.nonzero(predicted_labels)[0]
    
    # 根据索引获取相应的值
    true_labels = true_labels[nonzero_index]
    predicted_labels = predicted_labels[nonzero_index]

    # 计算预测正确的数量
    correct_count = np.sum(true_labels == predicted_labels)
    
    class_acc = class_accuracy(true_labels, predicted_labels, num_classes=num_classes)

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
    sum_acc = 0
    sum_class_acc = np.zeros(num_classes)
    count = 0
    
 
    
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
        
    
        merged_masks_transposed_0, merged_masks_0 = get_img_label(predictor, image_0, label_0, instance_label_0, pos2neg, config) # CMA_Front
        merged_masks_transposed_1, merged_masks_1 = get_img_label(predictor, image_1, label_1, instance_label_1, pos2neg, config) # CMA_Front_Left
        merged_masks_transposed_2, merged_masks_2 = get_img_label(predictor, image_2, label_2, instance_label_2, pos2neg, config) # CMA_Front_Right
        merged_masks_transposed_3, merged_masks_3 = get_img_label(predictor, image_3, label_3, instance_label_3, pos2neg, config) # CMA_Back
        merged_masks_transposed_4, merged_masks_4 = get_img_label(predictor, image_4, label_4, instance_label_4, pos2neg, config) # CMA_Back_Left
        merged_masks_transposed_5, merged_masks_5 = get_img_label(predictor, image_5, label_5, instance_label_5, pos2neg, config) # CMA_Back_Right
        
        if config.pseudo_type == "image":
            path = batch['path'][0]         
            root = config.root
            basename = os.path.basename(path).split('.')[0]  
            img_filename = os.path.join(root,  basename + ".npz") 
            
            np.savez_compressed(img_filename, 
                    idx_0=merged_masks_0.astype(np.uint8), 
                    idx_1=merged_masks_1.astype(np.uint8), 
                    idx_2=merged_masks_2.astype(np.uint8), 
                    idx_3=merged_masks_3.astype(np.uint8),
                    idx_4=merged_masks_4.astype(np.uint8),
                    idx_5=merged_masks_5.astype(np.uint8))
        
        if config.pseudo_type == "point":
            proj_labels_0 = merged_masks_transposed_0[img_indices_0[0][:, 0], img_indices_0[0][:, 1]]
            proj_labels_1 = merged_masks_transposed_1[img_indices_1[0][:, 0], img_indices_1[0][:, 1]]
            proj_labels_2 = merged_masks_transposed_2[img_indices_2[0][:, 0], img_indices_2[0][:, 1]]
            proj_labels_3 = merged_masks_transposed_3[img_indices_3[0][:, 0], img_indices_3[0][:, 1]]
            proj_labels_4 = merged_masks_transposed_4[img_indices_4[0][:, 0], img_indices_4[0][:, 1]]
            proj_labels_5 = merged_masks_transposed_5[img_indices_5[0][:, 0], img_indices_5[0][:, 1]]
        
            
            labels_0 = np.zeros(points.shape[0], dtype=np.int32)
            labels_0[p2img_idx_0[0]] = proj_labels_0[:,0] 

            labels_1 = np.zeros(points.shape[0], dtype=np.int32)
            labels_1[p2img_idx_1[0]] = proj_labels_1[:,0] 
            
            labels_2 = np.zeros(points.shape[0], dtype=np.int32)
            labels_2[p2img_idx_2[0]] = proj_labels_2[:,0]   
            
            labels_3 = np.zeros(points.shape[0], dtype=np.int32)
            labels_3[p2img_idx_3[0]] = proj_labels_3[:,0]
                    
            labels_4 = np.zeros(points.shape[0], dtype=np.int32)
            labels_4[p2img_idx_4[0]] = proj_labels_4[:,0] 
            
            labels_5 = np.zeros(points.shape[0], dtype=np.int32)
            labels_5[p2img_idx_5[0]] = proj_labels_5[:,0] 
            
            stack_label_1 = np.stack((labels_0, labels_1), axis=0)
            pseudo_labels_1 = majority_vote(stack_label_1) 

            stack_label_2 = np.stack((labels_0, labels_1, labels_2), axis=0)
            pseudo_labels_2 = majority_vote(stack_label_2) 
            
            stack_label_3 = np.stack((labels_0, labels_1, labels_2, labels_3), axis=0)
            pseudo_labels_3 = majority_vote(stack_label_3)        
            
            stack_label_4 = np.stack((labels_0, labels_1, labels_2, labels_3, labels_4), axis=0)
            pseudo_labels_4 = majority_vote(stack_label_4)                
            
            stack_label_5 = np.stack((labels_0, labels_1, labels_2, labels_3, labels_4, labels_5), axis=0)
            pseudo_labels_5 = majority_vote(stack_label_5)
            
            
            path = batch['path'][0]
            root = config.root
            if not os.path.exists(root+'val_pred_spvcnn'):
                    os.mkdir(root+'val_pred_spvcnn')              
            basename = os.path.basename(path).split('.')[0]
            
            if config.pseudo_labels_vis:
                filename = os.path.join(root, 'val_pred_spvcnn', basename + ".ply") 
                colorMap = np.array(config.colorMap, dtype=np.int32)
                labeled_point2ply(points, labels,filename,colorMap)     
            
               
            filename = os.path.join(root, 'val_pred_spvcnn', basename + ".npz")                     

            np.savez(filename, 
                    label_0=labels_0.astype(np.uint8), 
                    label_1=pseudo_labels_1.astype(np.uint8), 
                    label_2=pseudo_labels_2.astype(np.uint8), 
                    label_3=pseudo_labels_3.astype(np.uint8),
                    label_4=pseudo_labels_4.astype(np.uint8),
                    label_5=pseudo_labels_5.astype(np.uint8))
        
            # 统计pseudo_labels准确性
            accuracy,class_acc = compute_accuracy(labels.numpy(), pseudo_labels_5,num_classes)
            count += 1
            sum_acc += accuracy
            sum_class_acc += class_acc
            
    if config.pseudo_type == "point":        
        sum_result = sum_acc/count
        sum_class_result = sum_class_acc/count
        np.save('sum_class_acc.npy', sum_class_result) 
        print(sum_result)
        print(sum_class_result)