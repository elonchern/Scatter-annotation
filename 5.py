import matplotlib.pyplot as plt
import skimage.io as skio
from skimage.segmentation import slic
from skimage.color import label2rgb
from skimage.io import imsave

# 指定图像路径
input_image_path = "/data/elon/semantic_kitti/sequences/00/image_2/000000.png"  # 请替换为实际图像的路径

# 加载图像
image = skio.imread(input_image_path)

# 使用SLIC算法来分割图像
segments = slic(image, n_segments=200, compactness=50, start_label=1)

# 将分割结果可视化
segmented_image = label2rgb(segments, image, kind='avg')

# 保存分割后的图像为JPG文件
output_image_path = "/data/elon/val_pred_spvcnn/segmented_image.jpg"  # 指定保存路径
imsave(output_image_path, segmented_image)

# 可视化分割结果
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.imshow(segmented_image)
ax.set_title('Segmented Image')
ax.axis('off')

plt.show()  # 显示图像
