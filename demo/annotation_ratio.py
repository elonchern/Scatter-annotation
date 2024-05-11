import matplotlib.pyplot as plt

# 数据
x_labels = ['SLidR', 'IUPC', 'LESS', 'contra', 'ours']
y_values = [0.8, 0.8, 0.2, 0.2, 0.02]

# 设置每个柱子的位置
x_positions = [1, 2, 3, 4, 5]

colors = [(117, 164, 201),  
         (117, 164, 201),  
         (117, 164, 201),  
         (117, 164, 201),  
         (157, 105, 177)] 

# 将颜色值标准化到0到1的范围内
normalized_colors = [(r/255, g/255, b/255) for r, g, b in colors]
# 绘制柱状图
plt.bar(x_positions, y_values, width=0.6, color=normalized_colors)

# 设置刻度标签
plt.xticks(x_positions, x_labels)

# 添加标题和标签
plt.xlabel('Category')
plt.ylabel('Value')
plt.title('Bar Chart')

# Set y-axis limit to start from 0.01
plt.ylim(0.01, 0.85)

# Save the plot as an image
plt.savefig('../work_dirs/bar_chart.png')

# Show the plot
plt.show()
