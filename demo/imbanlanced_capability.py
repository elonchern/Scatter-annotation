import matplotlib.pyplot as plt

# 数据
x_labels = ['Label 1', 'Label 2', 'Label 3', 'Label 4', 'Label 5', 'Label 6']
y_data = [67.6, 56.44, 62.2, 63.8, 47.8, 57.4]

# 设置每个柱子的位置
x_positions = [1, 1.8, 2.6, 4.0, 4.8, 5.6]

colors = [(178, 178, 178),  # 红色
         (194, 220, 191),  # 蓝色
         (186,216, 251),  # 绿色
         (178, 178, 178),  # 橙色
         (194, 220,191),  # 紫色
         (186,216, 251)]  # 粉色

# 将颜色值标准化到0到1的范围内
normalized_colors = [(r/255, g/255, b/255) for r, g, b in colors]
# 绘制柱状图
plt.bar(x_positions, y_data, width=0.6, color=normalized_colors)

# 设置刻度标签
plt.xticks(x_positions, x_labels)

# 添加标题和标签
plt.title('xxxx')
plt.xlabel('xxx')
plt.ylabel('xxx')
plt.ylim(40, 72)
plt.savefig('../work_dirs/imbalanced.png')

# 显示图形
plt.show()
