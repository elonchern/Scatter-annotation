import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 文件路径检查函数
def file_exists(filepath):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"无法找到文件: {filepath}")

# 检查点是否在图像范围内
def is_within_bounds(point, image_shape):
    x, y = point
    return 0 <= x < image_shape[1] and 0 <= y < image_shape[0]

# 标签对应的颜色映射
colorMap = np.array([[0, 0, 0],   # 0 "unlabeled", and others ignored 0
                    [100, 150, 245],    # 1 "car" 10   495
                    [100, 230, 245],      # 2 "bicycle" 11 575 [100, 230, 245]
                    [30, 60, 150],   # 3 "motorcycle" 15 棕色 240
                    [80, 30, 180],   # 4 "truck" 18 绛红 290
                    [100, 80, 250],    # 5 "other-vehicle" 20 红色 430
                    [255, 30, 30],   # 6 "person" 30 淡蓝色 315
                    [255,40,200],   # 7 "bicyclist" 31 淡紫色 [255,40,200]
                    [150, 30, 90],    # 8 "motorcyclist" 32 深紫色  270
                    [255, 0, 255],    # 9 "road" 40 浅紫色 510
                    [255, 150, 255],    # 10 "parking" 44 紫色 660
                    [75, 0, 75],   # 11 "sidewalk" 48 紫黑色
                    [175, 0, 75],   # 12 "other-ground" 49 深蓝色 250
                    [255, 200, 0],   # 13 "building" 50 浅蓝色 455
                    [255, 120, 50],   # 14 "fence" 51 蓝色 425
                    [0, 175, 0],   # 15 "vegetation" 70 绿色175
                    [135, 60, 0],   # 16 "trunk" 71 蓝色 195
                    [150, 240, 80],   # 17 "terrain" 72 青绿色 470
                    [255, 240, 150],   # 18 "pole"80 天空蓝 645
                    [255, 0, 0]   # 19 "traffic-sign" 81 标准蓝
                    ]).astype(np.int32) 

# 初始要跟踪的点的坐标和标签
scatter_kitti_path = '/data/xzy/elon/seg/000290.npy'
file_exists(scatter_kitti_path)
scatter_kitti_data = np.load(scatter_kitti_path)
initial_points = np.argwhere(scatter_kitti_data != 0).astype(np.float32)

labels = np.array([scatter_kitti_data[int(point[0]), int(point[1])] for point in initial_points], dtype=np.uint8)


# 光流文件路径列表
flow_paths = [
    '/data/xzy/elon/seg/000290_flow.npy',
    '/data/xzy/elon/seg/000291_flow.npy',
    '/data/xzy/elon/seg/000292_flow.npy',
    '/data/xzy/elon/seg/000293_flow.npy'
]

# 图像文件路径列表
image_paths = [
    '/data/xzy/elon/JS3CNet/semantic_kitti/sequences/08/image_2/000291.png',
    '/data/xzy/elon/JS3CNet/semantic_kitti/sequences/08/image_2/000292.png',
    '/data/xzy/elon/JS3CNet/semantic_kitti/sequences/08/image_2/000293.png',
    '/data/xzy/elon/JS3CNet/semantic_kitti/sequences/08/image_2/000294.png'
]

# 确保光流文件和图像文件的数量相等
assert len(flow_paths) == len(image_paths), "光流文件和图像文件的数量不一致"

# 当前跟踪的点和标签
current_points = initial_points.copy()
current_labels = labels.copy()

# 遍历光流和图像
for i, (flow_path, image_path) in enumerate(zip(flow_paths, image_paths)):
    # 检查文件路径
    file_exists(flow_path)
    file_exists(image_path)

    # 加载光流数据
    flow = np.load(flow_path)

    # 加载对应的图像
    image = cv2.imread(image_path)

    # 确保图像成功加载
    if image is None:
        raise FileNotFoundError(f"无法加载图像: {image_path}")

    # 移除不在图像范围内的点和对应标签
    in_bounds_indices = [
        idx for idx, pt in enumerate(current_points) 
        if is_within_bounds((pt[0], pt[1]), image.shape)
    ]
    
    current_points = current_points[in_bounds_indices]
    current_labels = current_labels[in_bounds_indices]

    # 计算这些点在当前光流中的位移
    displacements = flow[current_points[:, 1].astype(int), current_points[:, 0].astype(int)]

    # 更新点的位置
    current_points += displacements

    # 在图像上绘制更新后的点
    for idx, position in enumerate(current_points):
        x, y = map(int, position)
        # 确保在图像范围内
        if is_within_bounds((x, y), image.shape):
            label = current_labels[idx]  # 对应的标签
            color = colorMap[label].tolist()  # 标签对应的颜色
            cv2.circle(image, (x+20, y), 10, color[::-1], -1)  # 使用标签对应的颜色绘制

    # 可视化图像
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f"跟踪点在帧 {i + 2} 上的可视化")
    plt.show()

    # 保存结果
    output_path = f'second_frame_with_tracked_points_{i + 2}.png'
    cv2.imwrite(output_path, image)
    print(f"结果图像已保存到: {output_path}")
