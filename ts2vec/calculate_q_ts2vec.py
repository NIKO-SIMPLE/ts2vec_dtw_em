import numpy as np
import pandas as pd
from ts2vec import TS2Vec

def calculate_q_ts2vec(Datas, demoNum, demoLen, alpha_n):
    """
        计算ts2vec的分数
        返回矩阵，矩阵i,j元表示第i条轨迹在第j个时间步上的评分
    """
    N = demoNum
    L = demoLen

    # 1. 数据重构与预处理
    """
    trajectories = []
    for i in range(N):
        start_col = i * L
        end_col = (i + 1) * L

        # 兼容一维和二维数组输入
        raw_data = Datas[0, start_col:end_col] if Datas.ndim == 2 else Datas[start_col:end_col]

        # 去均值中心化
        centered_data = raw_data - np.mean(raw_data)
        trajectories.append(centered_data)
    """

    data_array = np.array(Datas.reshape(demoNum,demoLen))

    train_data = data_array[..., None]

    model = TS2Vec(
        input_dims=1,
        device=0,  # 如果没有GPU，请使用 'cpu'
        output_dims=320
    )
    loss_log = model.fit(
        train_data,
        verbose=True
    )
    # 4.1 提取每个时间步的特征向量
    # 不指定 encoding_window，返回形状为 (N, L, 320) 的数组
    timestamp_repr = model.encode(train_data)
    print(timestamp_repr.shape)
    total_sum = np.sum(timestamp_repr, axis=0)
    print(total_sum.shape)

    # 2. "全加起来再减掉自己" -> 得到除 i 以外其他轨迹的和
    # 形状: (N, L, D)
    sum_others = total_sum[np.newaxis, :, :] - timestamp_repr
    print(sum_others.shape)
    # 3. 计算其他轨迹的平均值 (Mean of others)
    # 注意：分母是 N-1
    mean_others = sum_others / (N - 1)

    # 4. 计算相对局部语义偏差 n_{i,t} (公式第一行)
    # 即：当前特征 - 其他特征的均值，然后求 L2 范数
    diff = timestamp_repr - mean_others
    n_it = np.linalg.norm(diff, axis=2)  # 结果形状: (N, L)

    # 5. 计算 Softmax 归一化 (公式第二行)
    # 这里需要先对 n_it 在每一列（每个时间步 t）进行标准化 (z-score)
    mu_nt = np.mean(n_it, axis=0, keepdims=True)  # (1, L)
    sigma_nt = np.std(n_it, axis=0, keepdims=True) + 1e-8  # (1, L), 加 epsilon 防止除零

    # 标准化后的偏差
    z_score = (n_it - mu_nt) / sigma_nt

    # 应用 Softmax (注意 alpha_n 是超参数，通常取正数)
    # exp(-alpha * z)
    exponent = -alpha_n * z_score

    # 为了数值稳定性，通常减去每列的最大值
    exp_vals = np.exp(exponent)

    # 最终评分 q
    q_ts2vec = exp_vals / np.sum(exp_vals, axis=0, keepdims=True)
    """
    # 4.2 计算评分
    # 这里我们计算每个时间步特征向量的L2范数（欧几里得长度）作为评分
    # 范数越大，表示该时间步的特征越“显著”或“复杂”
    # 结果形状为 (N, L)
    scores = np.linalg.norm(timestamp_repr, axis=2)

    print(scores)
    """

    return q_ts2vec


if __name__ == "__main__":
    # 1. 准备测试数据 (模拟 12345.txt 的结构: 1行数据，包含10条长度为50的轨迹)
    np.random.seed(42)  # 固定随机种子，保证每次运行结果一致
    demoNum = 10  # 轨迹数量
    demoLen = 50  # 每条轨迹的长度
    alpha_n = 1.0  # 缩放系数 (预留参数)
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
    score_matrix = calculate_q_ts2vec(Datas, demoNum, demoLen, alpha_n)

