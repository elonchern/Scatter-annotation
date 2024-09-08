import matplotlib.pyplot as plt
from matplotlib import rcParams

# 定义横坐标
x_values = list(range(1, 11))

# 定义两条折线的纵坐标
# y_values1 = [89.5, 89.4, 89.3, 89.4, 89.5, 89.5, 87.5, 86.6, 84.2, 82.2]
# y_values2 = [85.5, 85.8, 85.0, 85.4, 85.5, 84.5, 84.5, 83.6, 78.2, 76.2]

y_values1 = [0.11, 0.12, 0.13, 0.16, 0.28, 0.64, 1.04, 1.59, 2.30, 3.11]
y_values2 = [0.10, 0.13, 0.14, 0.15, 0.21, 0.40, 0.80, 1.28, 2.0, 3.00]


# 定义RGB颜色
color1 = [239/255, 138/255, 67/255]   # RGB: [239, 138, 67]
color2 = [72/255, 101/255, 169/255]   # RGB: [72, 101, 169]

# 设置全局字体为 DejaVu Serif 或 Georgia
rcParams['font.family'] = 'DejaVu Serif'  # 或者 'Georgia'

# 绘制两条折线，一条带有三角形标记的直线，另一条为虚线
plt.plot(x_values, y_values1, label='nuScenes', marker='o', markersize=9,linestyle='-', color=color1,linewidth=2)
plt.plot(x_values, y_values2, label='SemanticKITTI', marker='s',markersize=9,linestyle='--', color=color2,linewidth=2)

# 添加图例
plt.legend(prop={'size': 21})

# 设置横坐标和纵坐标字体
plt.xlabel('Number of frames', fontsize=12)
plt.ylabel('AEPE', fontsize=12)

# 设置刻度标签的字体
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# 设置图的边框粗细
for spine in plt.gca().spines.values():
    spine.set_linewidth(2)

# 显示图形
plt.show()

# 保存图形
plt.savefig('../work_dirs/figure.png', format='png', dpi=300)
