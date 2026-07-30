import pandas as pd
import numpy as np
from calculate_sit import S_calculator

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
K=pdfNum=10
lambda_s=10

#数据参数
N=demoNum=10
T=demoLen=50


df = pd.read_csv('../12345.txt', sep='\s+', header=None)
raw_data =  df.values.astype(np.float64)

#取时间轴并正则化时间步
time_scale = raw_data[0 , 0:demoLen]
#print(time_scale)
X = np.exp(-alpha_x / tau * (time_scale-1))
#print(X)

#X维度上的轨迹
F_x = raw_data[2]
F_x = F_x.reshape(N,T)
#print(F_x.shape)
#print(F_x)

S_model= S_calculator(alpha_m=0.5, alpha_n=0.5, lambda_c=0.5, alpha_c=0.8)


S_it = S_model.calculate_S(F_x)
print(S_it)
print(np.sum(S_it))
col_sums = np.sum(S_it, axis=0)
print(col_sums)  # 应该全为 1.0
print(np.allclose(col_sums, 1.0))  # True