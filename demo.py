import matplotlib.pyplot as plt

# 数据
no_count = 2000
medium_count = 100
high_count = 5

# ===== 第一张图：No vs With（环形图） =====
labels1 = ['No', 'With']
sizes1 = [no_count, medium_count + high_count]
colors1 = ['#66c2a5', '#fc8d62']

# ===== 第二张图：Medium vs High（柱状图） =====
labels2 = ['Medium', 'High']
values2 = [medium_count, high_count]
colors2 = ['#fc8d62', '#8da0cb']

# 画图
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# 左边环形图
wedges1, texts1, autotexts1 = axs[0].pie(
    sizes1,
    labels=labels1,
    autopct='%1.2f%%',
    startangle=90,
    colors=colors1,
    wedgeprops=dict(width=0.4)  # 控制环的粗细
)
axs[0].set_title("No vs With")
plt.setp(autotexts1, size=10, weight="bold")

# 右边柱状图
axs[1].bar(labels2, values2, color=colors2)
axs[1].set_title("Medium vs High (Count)")
axs[1].set_ylabel("Count")
for i, v in enumerate(values2):
    axs[1].text(i, v + 1, str(v), ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()
