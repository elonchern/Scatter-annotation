import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

# 定义类别和对应的数量
categories =  ['car', 'bicycle', 'motorcycle', 'truck','other-vehicle','person','bicyclist','motorcyclist','road','parking',
              'sidewalk','other-ground','building','fence','vegetation','trunk','terrain','pole','traffic-sign']
values = [0.91445354, 0.1802871,  0.07036507, 0.02046779, 0.17476557,
            0.19155636, 0.15934671, 0.01397761, 0.97317924, 0.22063157, 0.85305587,
            0.03485138, 0.75589207, 0.52576894, 0.77198219, 0.59244093, 0.67953983, 0.54317137, 0.32076339]

# 设置颜色和透明度
colors = [(100, 150, 245), (100, 230, 245), (30, 60, 15),(80, 30, 180),(100, 80, 250),(255, 30, 30),(255,40,200),(150, 30, 90),
          (255, 0, 250),(255, 150, 255),(75, 0, 75),(175, 0, 75),(255, 200, 0),(255, 120, 50),(0, 175, 0),(135, 60, 0),(150, 240, 80),(255, 240, 150),(255, 0, 0)]
alpha = [0.5, 0.5, 0.5,0.5,0.5,0.5, 0.5, 0.5,0.5,0.5,0.5, 0.5, 0.5,0.5,0.5,0.5,0.5,0.5,0.5]

# 自定义百分制格式
def to_percent(y, position):
    return "{:.0%}".format(y)

plt.figure(figsize=(20, 4))
# 处理其他类别
for i in range(0, len(categories)):
    plt.bar(categories[i], values[i], color=(colors[i][0]/255, colors[i][1]/255, colors[i][2]/255), alpha=alpha[i], width=0.4)

# 设置y轴的百分制格式
plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))

plt.show()



plt.savefig('../work_dirs/figure_3a.jpg', format='jpg', dpi=300)
# plt.show()
