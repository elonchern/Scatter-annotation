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
import multiprocessing as mp
from segment_anything import sam_model_registry, SamPredictor
from Mask2Former.mask2former import add_maskformer2_config
from Mask2Former.demo.predictor import VisualizationDemo
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
import argparse

def load_yaml(file_name):
    with open(file_name, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            config = yaml.load(f)
    return config


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg



import argparse
from easydict import EasyDict

def parse_config():
    parser = argparse.ArgumentParser(description="Combined configuration parser for scatter annotation and maskformer2 demo")
    
    # General configuration for scatter annotation
    parser.add_argument('--gpu', type=int, nargs='+', default=(0,), help='specify gpu devices')
    parser.add_argument("--seed", default=0, type=int, help="Random seed")
    parser.add_argument('--config_path', default='../../config/semantickitti.yaml', help="Path to the configuration file")
    parser.add_argument('--sam_checkpoint', type=str, default="../../checkpoint/sam_vit_h_4b8939.pth", help='Path to the SAM checkpoint file')
    parser.add_argument('--model_type', type=str, default="vit_h", help='Type of the model (e.g., vit_h)')
    parser.add_argument('--pseudo_type', type=str, default="point", help='Type of the pseudo (e.g., point, image)')
    parser.add_argument('--pseudo_labels_vis', type=bool, default=True, help="Whether to visualize pseudo labels")
    parser.add_argument('--device', type=str, default="cuda", help='Device to run the model on (e.g., cuda)')
    parser.add_argument('--root', type=str, default='/data/xzy/elon/mask2former2', help='The root directory to save pseudo labels')
    parser.add_argument('--debug', default=False, action='store_true', help='Enable debug mode')

    # Maskformer2 specific arguments
    parser.add_argument(
        "--config-file",
        default="configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k.yaml",
        metavar="FILE",
        help="Path to maskformer2 config file",
    )
    
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )

    # Parse arguments
    args = parser.parse_args()

    # Load YAML configuration for scatter and merge with command-line arguments
    config = load_yaml(args.config_path)
    config.update(vars(args))  # Override the configuration with command-line values

    # Return both EasyDict and argparse.Namespace
    return EasyDict(config), args



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
    config, args = parse_config()
    print(config)

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, config.gpu))
    num_gpu = len(config.gpu)
    


    # 加载Mask2Former模型
    mp.set_start_method("spawn", force=True)
    
    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)


       
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

    with open(config['dataset_params']['label_mapping'], 'r') as stream:
            semkittiyaml = yaml.safe_load(stream) 
    cityscapes_map = semkittiyaml['cityscapes_map']
    
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
        
        predictions, visualized_output = demo.run_on_image(image.cpu().numpy()[:, :, ::-1]) # image (np.ndarray): an image of shape (H, W, C) (in BGR order).[370,1220,3]

        # predictions -> Mask predictions['sem_seg']=[c,370,1220] -> mask=[1, 370, 1226] mapping
        predicted_segmentation = torch.argmax(predictions['sem_seg'], dim=0)

        # lable mapping cityspace to kitti

        merged_masks = np.vectorize(cityscapes_map.__getitem__)(predicted_segmentation.cpu().numpy()) # 函数向量化
        merged_masks = np.expand_dims(merged_masks, axis=0)
                         
        path = batch['path'][0]
        root = config.root # '/data/elon/segment/05/'
        basename = os.path.basename(path).split('.')[0] 
        if config.pseudo_type == "image":
            pseudo_img_filename = os.path.join(root,  basename + ".npy") 
            # np.save(pseudo_img_filename, merged_masks.astype(np.uint8))
            if config.pseudo_labels_vis:
                img_filename = os.path.join(root,  basename + ".jpg") 
                merged_masks_transposed = np.transpose(merged_masks, (1, 2, 0)) # [1, 370, 1226] -> [370, 1226, 1]
                colorMap = np.array(config.colorMap, dtype=np.int32)
                point2cam_label(merged_masks_transposed, image.cpu().permute(2, 0, 1), img_filename,colorMap)  
            
             
        if config.pseudo_type == "point":
            merged_masks_transposed = np.transpose(merged_masks, (1, 2, 0)) # [1, 370, 1226] -> [370, 1226, 1]   
            proj_labels = merged_masks_transposed[img_indices[0][:, 0], img_indices[0][:, 1]]
            # pseudo_labels = np.zeros(points.shape[0], dtype=np.int32)
            pseudo_labels = np.full(points.shape[0], -1, dtype=np.int32)
            pseudo_labels[p2img_idx[0]] = proj_labels[:,0]
            pseudo_point_filename = os.path.join(root, basename + ".npy") 
            # np.save(pseudo_point_filename, pseudo_labels.astype(np.uint8))
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
    
     
        
      