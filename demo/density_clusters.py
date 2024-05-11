import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN

# 生成一些示例数据
# # 你可以替换这个部分来加载你的实际点云数据
# N = 1000  # 点的数量
# points = np.random.rand(N, 3) * 100  # 生成随机点

raw_data = np.fromfile("/data/elon/semantic_kitti/sequences/00/velodyne/000000.bin", dtype=np.float32).reshape((-1, 4))
origin_len = len(raw_data)
points = raw_data[:, :3]
filtered_points1 = points[points[:, 2] >= -1.2]
filtered_points = filtered_points1[filtered_points1[:, 0] >= 0]

# 使用 DBSCAN 进行聚类
# eps 控制邻近距离，min_samples 是簇的最小样本数
clustering = DBSCAN(eps=1, min_samples=5).fit(filtered_points)

# 获取每个点的簇标签
labels = clustering.labels_

# 获取簇的数量（忽略噪音簇 -1）
num_clusters = len(set(labels)) - (1 if -1 in labels else 0)


# 为每个簇分配随机颜色
colors = np.random.rand(num_clusters, 3)  # 生成随机颜色

# 为点云中的每个点分配颜色
point_colors = np.array([colors[label] if label != -1 else [0, 0, 0] for label in labels])  # 噪音用黑色


# 创建 Open3D 点云对象
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(filtered_points)  # 设置点的坐标
point_cloud.colors = o3d.utility.Vector3dVector(point_colors)  # 设置点的颜色

# 保存为 PLY 文件
output_ply_path = "../work_dirs/clustered_point_cloud.ply"  # 保存路径
o3d.io.write_point_cloud(output_ply_path, point_cloud)
