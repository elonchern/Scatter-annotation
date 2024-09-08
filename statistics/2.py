import matplotlib.pyplot as plt

# 将 0-255 范围的 RGB 值转换为 0-1 范围
def rgb_to_mpl_color(r, g, b):
    return (r / 255.0, g / 255.0, b / 255.0)

# 定义散点坐标、形状、颜色（使用 0-255 的 RGB 值）和大小
points = [
    (0.8, 94, 'o', rgb_to_mpl_color(241, 158, 156), 100),       # 蓝色
    (0.8, 98.4, '^', rgb_to_mpl_color(177, 252, 163), 100),     # 绿色
    (0.2, 84.2, 's', rgb_to_mpl_color(207, 144, 236), 100),   # 橙色
    (0.2, 97.4, 'D', rgb_to_mpl_color(154,154,248), 100),   # 紫色
    (0.02, 98.0, '*', rgb_to_mpl_color(71, 88, 162), 200)     # 红色
]
# 创建图形和轴对象
fig, ax = plt.subplots()

# 绘制散点图
for x, y, marker, color, size in points:
    plt.scatter(x, y, marker=marker, color=color, edgecolor='black', s=size)

# 设置 x 轴范围为 0 到 1，y 轴范围为 80 到 100
plt.xlim(-0.05, 0.9)
plt.ylim(80, 100)

# 增加网格线
ax.grid(True, linestyle='--', linewidth=0.5, color='grey', alpha=0.7)

plt.xlabel('X轴标签')
plt.ylabel('Y轴标签')
plt.title('散点图')

# 设置图的边框粗细
for spine in plt.gca().spines.values():
    spine.set_linewidth(1.5)

plt.show()
plt.savefig('../work_dirs/figure.png', format='png', dpi=300)
