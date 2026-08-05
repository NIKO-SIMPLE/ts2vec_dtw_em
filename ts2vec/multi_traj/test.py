import pandas as pd
import numpy as np
from calculate_sit import S_calculator
from EM import EM


#正则系统参数
tau =1.0
alpha_x =1.0
x0 =1.0

#计算S_it所用参数
alpha_m=0.5
alpha_n=0.5
lambda_c=0.5
alpha_c=0.8

#最小化时关于平滑度的参数
K=pdfNum=25
lambda_s=20

#数据参数
N=demoNum=10
T=demoLen=50


df = pd.read_csv('../12345.txt', sep='\s+', header=None)
raw_data =  df.values.astype(np.float64)

#取时间轴并正则化时间步
time_scale = raw_data[0 , 0:demoLen]
#print(time_scale)
X = np.exp(-alpha_x / tau * (time_scale-1))
#print(X.shape)

#X维度上的轨迹
F_x = raw_data[2]
F_x = F_x.reshape(N,T)
#print(F_x.shape)
#print(F_x)

S_model= S_calculator(alpha_m=0.5, alpha_n=0.5, lambda_c=0.5, alpha_c=0.8)


S_it = S_model.calculate_S(F_x)

EM_model = EM(K,lambda_s)

EM_model.loading_Data(X, F_x, S_it)
print("开始 EM 训练 ...")
for epoch in range(100):
    EM_model.E_step()
    EM_model.M_step()

    # 每 10 轮打印一次进度（可选）
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{100} 完成")

print("训练结束！")

smooth_z=0.0
smooth_f=[0.0 for i in range(N)]
#计算平滑度
for i in range(T-1):
    smooth_z += (EM_model.Z[i+1]-EM_model.Z[i])**2
    for j in range(N):
        smooth_f[j] +=(F_x[j][i+1]-F_x[j][i]) ** 2

smooth_z /= T
for j in range(N):
    smooth_f[j] /= T

print(EM_model.Z)
print(F_x[0])

print(smooth_z)
print(smooth_f)


np.savetxt('Z_result.txt', EM_model.Z, fmt='%.6f')

import matplotlib.pyplot as plt

# 设置中文字体（避免中文标签乱码）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(12, 6))

# 1. 绘制 F_x：所有轨迹用半透明蓝色，展示数据分布
for i in range(N):
    # 第一条轨迹加label，其余不加避免图例重复
    label = 'F_x (观测轨迹)' if i == 0 else None
    ax.plot(time_scale, F_x[i], color='blue', alpha=0.3, linewidth=0.8, label=label)

# 2. 绘制 Z：红色实线，加粗突出
ax.plot(time_scale, EM_model.Z, color='red', linewidth=0.8, label='Z (隐变量)')

# 3. 图表修饰
ax.set_xlabel('时间', fontsize=12)
ax.set_ylabel('幅值', fontsize=12)
ax.set_title('观测轨迹 F_x 与隐变量 Z 对比', fontsize=14)
ax.legend(fontsize=11, loc='best')
ax.grid(True, linestyle='--', alpha=0.5)

# 4. 保存与显示
plt.tight_layout()
plt.savefig('F_x_Z_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("图片已保存为 F_x_Z_comparison.png")

