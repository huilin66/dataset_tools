import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# 示例医疗数据：不同指标的两个数据系列的数值
data = pd.DataFrame({
    'Indicator': ['P', 'R', 'mAP50', 'mAP50-95'],
    'yolo9': [0.957, 0.812, 0.875, 0.701],
    'yolo8': [0.937, 0.797, 0.88, 0.741],
    'tfd': [0.938, 0.82, 0.88, 0.706],
    'ema2': [0.952, 0.84, 0.887, 0.71],
    'ema3': [0.936, 0.852, 0.888, 0.711],
})

# 创建雷达图
plt.figure(figsize=(8, 8))
sns.set_style("whitegrid")

# 设置雷达图的角度和标签
categories = list(data['Indicator'])
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

# 绘制雷达图的轴线
ax = plt.subplot(111, polar=True)
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
plt.xticks(angles[:-1], categories, fontsize=12)
ax.set_ylim(0.7, 1)
# ax.set_yticks(np.arange(0, 1, 10))


# 使用亮色绘制两个数据系列
values_series1 = list(data['tfd'])
values_series1 += values_series1[:1]
ax.fill(angles, values_series1, 'm', alpha=0.3, label='Model 1')

values_series2 = list(data['ema2'])
values_series2 += values_series2[:1]
ax.fill(angles, values_series2, 'c', alpha=0.3, label='Model 2')

values_series3 = list(data['ema3'])
values_series3 += values_series3[:1]
ax.fill(angles, values_series3, 'r', alpha=0.3, label='Model 3')


values_series4 = list(data['yolo9'])
values_series4 += values_series4[:1]
ax.fill(angles, values_series4, 'g', alpha=0.3, label='Model 4')

values_series5 = list(data['yolo8'])
values_series5 += values_series5[:1]
ax.fill(angles, values_series5, 'b', alpha=0.3, label='Model 5')
# 添加标题和图例
plt.title("metrics comparison", fontsize=16)
plt.legend(loc='upper right', title="Data Series")

# 显示雷达图
plt.show()