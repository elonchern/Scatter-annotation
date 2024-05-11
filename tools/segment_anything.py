import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from easydict import EasyDict
from argparse import ArgumentParser
from tqdm import tqdm
from PIL import Image

def parse_config():
    parser = ArgumentParser()
    # general
    parser.add_argument('--gpu', type=int, nargs='+', default=(1,), help='specify gpu devices')
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument('--input_folder',type=str,default = '/data/elon/semantic_kitti/sequences/08/image_2',help='the root directory to save images')
    parser.add_argument('--output_folder',type=str,default = '/data/elon/val_pred_spvcnn',help='the root directory to save images')
    parser.add_argument('--sam_checkpoint', type=str, default="/home/elon/Workshops/Scatter-annotation/checkpoint/sam_vit_h_4b8939.pth",
                        help='Path to the SAM checkpoint file')
    parser.add_argument('--model_type', type=str, default="vit_h", help='Type of the model (e.g., vit_h)')
    parser.add_argument('--device', type=str, default="cuda", help='Device to run the model on (e.g., cuda)')
    
 
    args = parser.parse_args()
    config = EasyDict(vars(args))
   

    return config


if __name__ == '__main__':
# 获取配置
    config = parse_config()

    # 文件夹路径
    input_folder = config.input_folder
    output_folder = config.output_folder

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # SAM 模型设置
    model_type = config.model_type
    checkpoint_path = config.sam_checkpoint
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=config.device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    # 获取文件夹中所有图片的路径
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort()

    # 使用 tqdm 添加进度条
    for image_file in tqdm(image_files, desc="Processing images"):
        # 加载图像
        image_path = os.path.join(input_folder, image_file)
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 使用 PIL 将 OpenCV 图像转换为 Image 对象
        pil_image = Image.fromarray(image_rgb)

        # 生成掩码
        overlay = np.zeros_like(image_rgb, dtype=np.uint8)  # 确保掩码的类型是 uint8
        masks = mask_generator.generate(image_rgb)

        # 为每个掩码分配随机颜色
        for mask in masks:
            color = (np.array(np.random.rand(3)) * 255).astype(np.uint8)  # 将颜色转换为 uint8
            for i in range(3):
                # 应用掩码时，确保类型匹配
                overlay[..., i] = np.where(mask['segmentation'], color[i], overlay[..., i])

        # 将掩码转换为 PIL 图像
        pil_overlay = Image.fromarray(overlay)

        # 使用 PIL 的 blend 方法混合图像
        alpha = 0.7  # 混合比例
        blended_image = Image.blend(pil_image, pil_overlay, alpha)

        # 保存混合后的图像
        output_image_path = os.path.join(output_folder, "seg_" + image_file)
        blended_image.save(output_image_path, "JPEG")
