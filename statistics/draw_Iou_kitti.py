import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

# 定义类别和对应的数量
categories =  ['car', 'bicycle', 'motorcycle', 'truck','other-vehicle','person','bicyclist','motorcyclist','road','parking',
              'sidewalk','other-ground','building','fence','vegetation','trunk','terrain','pole','traffic-sign']
values = [0.8915, 0.3012,  0.4260, 0.7017, 0.6625,
            0.3404, 0.7204, 0.4604, 0.9558, 0.6307, 0.7987,
            0.6975, 0.7095, 0.5093, 0.8397, 0.4632, 0.8482, 0.5760, 0.4076]

# 设置颜色和透明度
colors = [(100, 150, 245), (100, 230, 245), (30, 60, 15),(80, 30, 180),(100, 80, 250),(255, 30, 30),(255,40,200),(150, 30, 90),
          (255, 0, 250),(255, 150, 255),(75, 0, 75),(175, 0, 75),(255, 200, 0),(255, 120, 50),(0, 175, 0),(135, 60, 0),(150, 240, 80),(255, 240, 150),(255, 0, 0)]
alpha = [1.0, 1.0, 1.0,1.0,1.0,1.0, 1.0, 1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]

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



plt.savefig('../work_dirs/iou_kitti.jpg', format='jpg', dpi=300)
# plt.show()
