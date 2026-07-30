import numpy as np
import pandas as pd
from ts2vec import TS2Vec
from calculate_q_dtw import calculate_q_DTW
from calculate_q_ts2vec import calculate_q_ts2vec
def calculate_s_it(Datas, demoNum, demoLen, alpha_m, alpha_n, alpha_c, lambda_c):

    q_i_DTW=calculate_q_DTW(Datas,demoNum,demoLen,alpha_m)
    q_ts2vec=calculate_q_ts2vec(Datas,demoNum,demoLen,alpha_n)

    term_dtw = np.power(q_i_DTW[:, np.newaxis], lambda_c)
    term_ts2vec = np.power(q_ts2vec, 1.0 - lambda_c)

    # 组合得分矩阵 Score_{i,t}，形状为 (N, L)
    combined_scores = alpha_c * term_dtw * term_ts2vec

    # 3. 计算 Softmax (不做最大值减法)
    # 对每个时间步 t (列)，计算所有样本 i (行) 的指数和
    exp_scores = np.exp(combined_scores)

    # 分母求和: 沿着 axis=0 (样本维度) 求和，保持维度以便广播除法
    sum_exp_scores = np.sum(exp_scores, axis=0, keepdims=True)

    # 4. 计算最终权重 s_{i,t}
    s_it = exp_scores / sum_exp_scores
    #print(s_it.shape)
    #print(s_it)
    return s_it

if __name__ == "__main__":
    # 1. 准备测试数据 (模拟 12345.txt 的结构: 1行数据，包含10条长度为50的轨迹)
    np.random.seed(42)  # 固定随机种子，保证每次运行结果一致
    demoNum = 10  # 轨迹数量
    demoLen = 50  # 每条轨迹的长度
    alpha_m = 1.0
    alpha_n = 1.0  # 缩放系数 (预留参数)
    alpha_c = 1.0
    lambda_c = 0.5
    df = pd.read_csv('12345.txt', sep='\s+', header=None, skiprows=2, nrows=1)
    raw_data = df.values.astype(np.float64)
    # 生成 9 条正常的正弦波轨迹（带一点随机噪声）
    t = np.linspace(0, 2 * np.pi, demoLen)
    normal_trajs = np.sin(t) + np.random.normal(0, 0.05, (9, demoLen))

    # 生成 1 条异常轨迹（形状差异较大，例如频率加倍且振幅变大）
    abnormal_traj = np.sin(2 * t) * 3 + np.random.normal(0, 0.1, demoLen)

    # 拼接成 MATLAB 格式的宽矩阵 [1 x (demoNum * demoLen)]
    all_trajs = df.values.astype(np.float64)
    Datas = all_trajs.reshape(1, -1)

    # 2. 调用函数计算评分矩阵
    score_matrix = calculate_s_it(Datas, demoNum, demoLen,alpha_m, alpha_n,alpha_c, lambda_c)






