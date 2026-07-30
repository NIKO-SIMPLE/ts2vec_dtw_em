import numpy as np
from dtw import dtw
import sys
import os

# 获取 multi_traj 的父目录（即 ts2vec.py 所在目录）
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# parent_dir = C:\Users\16431\Desktop\ts2vec\ts2vec

# 添加到系统路径
sys.path.insert(0, parent_dir)

# 直接导入 ts2vec（因为 ts2vec.py 就在 parent_dir 下）
from ts2vec import TS2Vec

class S_calculator:
    def __init__(self, alpha_m=0.5, alpha_n=0.5, lambda_c=0.5, alpha_c=0.8):
        self.alpha_m=alpha_m
        self.alpha_n=alpha_n
        self.alpha_c=alpha_c
        self.lambda_c=lambda_c

    """计算DTW评分"""

    def calculate_DTW(self, F_x):
        #1. 取维度
        N,T=F_x.shape

        #2. 对所有轨迹做去均值中心化处理
        mean_F_x = np.mean(F_x, axis=0)
        F_x_centered = F_x - mean_F_x

        #3. 依次对每个轨迹求m
        m = np.zeros(N)
        for i in range(N):
            dist_sum = 0
            for j in range(N):
                if i == j:
                    continue
                else:
                    alignment = dtw(F_x_centered[i],F_x_centered[j], dist_method='euclidean')
                    d = alignment.distance
                    dist_sum += d
            m[i] = 1/(N-1) * dist_sum

        m_mean = np.mean(m)
        m_std =  np.std(m)

        exp = np.zeros(N)
        for i in range(N):
            exp[i] = np.exp( -self.alpha_m * (m[i] - m_mean) / m_std)

        exp_sum = np.sum(exp)
        q_DTW = exp / exp_sum

        return q_DTW

    """计算ts2vec评分"""

    def calculate_ts2vec(self, F_x):
        #1. 取维度
        N,T=F_x.shape

        #2. 对所有轨迹做去均值中心化处理,并转换为三维格式应用ts2vec
        mean_F_x = np.mean(F_x, axis=0)
        F_x_centered = F_x - mean_F_x

        train_data = F_x_centered.reshape(N, T, 1)

        model = TS2Vec(
            input_dims=1,  # 1维特征
            device=0,  # 使用 GPU 0，如果没有 GPU 可设为 -1
            output_dims=320  # 输出维度，即将轨迹转成多少维向量
        )

        loss_log = model.fit(
            train_data,
            verbose=True
        )

        # 4. 获取实例级表征 (每个轨迹一个向量)
        code = model.encode(
            train_data,
        )  #N x T x output_dims

        #5. 计算得分
        code_sum = np.sum(code, axis=0)
        code_sum_without_self = (code_sum - code) / (N - 1)

        diff = code -code_sum_without_self
        n = np.linalg.norm(diff, axis=2) #N x T

        #6. 归一化
        n_mean = np.mean(n,axis=0) #T x 1
        n_std = np.std(n,axis=0)  #T x 1
        n_std = np.where(n_std == 0, 1e-10, n_std)

        exp = np.exp(-self.alpha_n * (n - n_mean) / n_std)  # shape: (N, T)

        exp_sum = np.sum(exp,axis=0)
        q_ts2 = exp / exp_sum

        return q_ts2

    """计算总分"""
    def calculate_S(self, F_x):
        #1. 首先计算DTW评分和ts2评分
        q_DTW = self.calculate_DTW(F_x)
        q_ts2 = self.calculate_ts2vec(F_x)

        #2. 计算指数
        exp = np.exp(self.alpha_c * (q_DTW[:, np.newaxis] ** self.lambda_c) * (q_ts2 ** ( 1.0 - self.lambda_c ))) #N x T
        exp_sum = np.sum(exp,axis=0) #T x 1

        s_it= exp / exp_sum
        return s_it

