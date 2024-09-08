import matplotlib.pyplot as plt
from matplotlib import rcParams
# 设置全局字体为 DejaVu Serif 或 Georgia
rcParams['font.family'] = 'DejaVu Serif'  # 或者 'Georgia'
# 横坐标数据
x = [10, 20, 30, 40, 50, 60, 70]

# 纵坐标数据
y1 = [86, 90, 86, 76, 61, 20, 12]
y2 = [88, 89, 84, 70, 60, 21, 11]
y3 = [65, 63, 58, 52, 45, 21, 10]
y4 = [61, 62, 60, 50, 40, 9, 10]

# RGB颜色定义，范围0-255
color1 = '#797BB7'  # 紫色
color2 = '#B595BF'  # 橙色
color3 = '#F7D58B'  # 蓝绿色
color4 = '#9BC985'  # 金色

# 绘制第一条线
plt.plot(x, y1, color=color1, linestyle='-', marker='o', label='MM-ScatterNet')
# 绘制第二条线
plt.plot(x, y2, color=color2, linestyle='--', marker='s', label='2DPASS')
# 绘制第三条线
plt.plot(x, y3, color=color3, linestyle='-.', marker='^', label='PMF')
# 绘制第四条线
plt.plot(x, y4, color=color4, linestyle=':', marker='*', label='Baseline')

# 设置 x 轴范围为 0 到 1，y 轴范围为 80 到 100
plt.xlim(5, 75)
plt.ylim(0, 100)

# 添加标题和标签
# plt.title('折线图示例')
plt.xlabel('横坐标')
plt.ylabel('纵坐标')


# 添加图例
plt.legend(prop={'size': 13})

# 设置横坐标和纵坐标字体
plt.xlabel('Distance(m)', fontsize=13)
plt.ylabel('mIoU(%)', fontsize=13)

# 设置刻度标签的字体
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# 设置图的边框粗细
for spine in plt.gca().spines.values():
    spine.set_linewidth(2)
# 显示网格
# plt.grid(True)

# 显示图形
plt.show()

# 保存图形
plt.savefig('../work_dirs/figure.png', format='png', dpi=300)