import matplotlib.pyplot as plt
from matplotlib import rcParams
# 设置全局字体为 DejaVu Serif 或 Georgia
rcParams['font.family'] = 'DejaVu Serif'  # 或者 'Georgia'
# 横坐标
x = [1, 2, 3, 4, 5, 6]
# 纵坐标
y = [58.47, 69.5, 70.6, 74.2, 75.45, 76]

# 新的线的纵坐标
y_new = [76.3, 76.6, 76.9, 77.0, 77.5, 77.9]

# 绘制折线图
plt.plot(x, y, marker='o', markersize=9, color=(72/255, 101/255, 169/255), label='w removing')  # 使用 RGB 颜色
plt.plot(x, y_new, marker='^', markersize=9, linestyle='--', color=(255/255, 99/255, 71/255), label='w/o removing')  # 新的线，使用 RGB 颜色

# 添加直线
plt.plot([1, 6], [80, 80], color=(72/255, 101/255, 169/255), linewidth=2)  # 使用 RGB 颜色

# 设置 x 轴范围为 0 到 7，y 轴范围为 50 到 90
plt.xlim(0.8, 6.5)
plt.ylim(57.5, 79)

# 添加标题和标签
# plt.title('xxx')
plt.xlabel('Number of cameras',fontsize=13)
plt.ylabel('mIoU (%)',fontsize=13)

plt.grid(axis='y', color='gray', linestyle='--', linewidth=0.5)

# 添加图例
plt.legend(prop={'size': 13})

# 设置刻度标签的字体
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)

# 设置图的边框粗细
for spine in plt.gca().spines.values():
    spine.set_linewidth(2)
# 显示图形
plt.show()

# 保存图形
plt.savefig('../work_dirs/figure.png', format='png', dpi=300)
