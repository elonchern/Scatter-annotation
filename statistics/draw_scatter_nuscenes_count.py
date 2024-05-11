import matplotlib.pyplot as plt
import numpy as np

# 定义类别和对应的数量
categories = ['barrier', 'bicycle', 'bus', 'car','construction_vehicle','motorcycle','pedestrian','traffic_cone',
              'trailer','truck','driveable_surface','other_flat','sidewalk','terrain','manmade','vegetation']
values_x = [9.30510600e+06, 1.41351000e+05, 4.60476000e+06, 3.81042190e+07, 1.51441400e+06, 
            4.27391000e+05, 2.31472700e+06,7.36239000e+05, 4.90751100e+06, 1.58413840e+07, 
            3.16958899e+08,8.55921600e+06, 7.01974610e+07, 7.02897300e+07, 1.78178063e+08,1.22581273e+08]
values_y = [1.05958000e+04, 2.78450000e+03, 5.42650000e+03,8.16360800e+04, 4.85660000e+03, 
            1.24780000e+03, 4.11735000e+04,1.34983000e+04, 6.15160000e+03, 3.11545000e+04, 
            1.24900200e+05, 2.72429000e+04, 1.14866800e+05, 9.29756300e+04, 1.41722400e+05, 1.19562300e+05]

# 设置颜色和透明度
colors = [(255, 120, 50), (100, 230, 245), (135, 60, 0),(100, 150, 245),(100, 80, 250),(30, 60, 150),(255, 30, 30),(255, 124, 128),
          (255, 240, 150),(80, 30, 180),(255, 0, 255),(175, 0, 75),(75, 0, 75),(150, 240, 80),(255, 200, 0),(0, 175, 0)]
alpha_x = [0.5, 0.5, 0.5,0.5,0.5,0.5, 0.5, 0.5,0.5,0.5,0.5, 0.5, 0.5,0.5,0.5,0.5]
alpha_y = [0.5, 0.5, 0.5,0.5,0.5,0.5, 0.5, 0.5,0.5,0.5,0.5, 0.5, 0.5,0.5,0.5,0.5]


plt.figure(figsize=(20, 4))
# 处理其他类别
for i in range(0, len(categories)):
    
    plt.bar(categories[i], values_x[i], color=(colors[i][0]/255, colors[i][1]/255, colors[i][2]/255), alpha=alpha_x[i], width=0.4,label='数据集 X')
    plt.bar(categories[i], values_y[i], color=(colors[i][0]/255, colors[i][1]/255, colors[i][2]/255), alpha=alpha_y[i], width=0.4,label='数据集 Y')

# 设置y轴为对数尺度
plt.yscale('log')
plt.gca().set_ylim([1, max(values_x) * 10])

# 添加标题和标签
# plt.title('柱状图')
# plt.xlabel('类别')
# plt.ylabel('数量')
# plt.legend() # 添加图例

# 显示图形
plt.show()



plt.savefig('../work_dirs/figure_3b.jpg', format='jpg', dpi=300)
# plt.show()
