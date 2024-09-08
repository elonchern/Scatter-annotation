import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

# 定义类别和对应的数量
categories = ['barrier', 'bicycle', 'bus', 'car','construction_vehicle','motorcycle','pedestrian','traffic_cone',
              'trailer','truck','driveable_surface','other_flat','sidewalk','terrain','manmade','vegetation']
values = [0.5674, 0.5789, 0.7975, 0.9402, 0.4808, 0.5423, 0.6019, 0.4558, 0.5774, 0.8427, 0.9329, 0.8393, 0.7958, 0.8214, 0.8502, 0.8017]

# 设置颜色和透明度
colors = [(255, 120, 50), (100, 230, 245), (135, 60, 0),(100, 150, 245),(100, 80, 250),(30, 60, 150),(255, 30, 30),(255, 124, 128),
          (255, 240, 150),(80, 30, 180),(255, 0, 255),(175, 0, 75),(75, 0, 75),(150, 240, 80),(255, 200, 0),(0, 175, 0)]
alpha = [1.0, 1.0, 1.0,1.0,1.0,1.0, 1.0, 1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]

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

plt.savefig('../work_dirs/Iou_nus.jpg', format='jpg', dpi=300)
# plt.show()
