import numpy as np
from dtw import dtw


def calculate_q_DTW(Datas, demoNum, demoLen, alpha_m):
    """
    计算每个演示数据的 DTW 质量分数
    """
    N = demoNum
    L = demoLen
    """
    # 1. 数据重构与预处理
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
    trajectories = np.array(Datas.reshape(demoNum, demoLen))

    # 2. 计算平均全局偏差 m_i
    m_values = np.zeros(N)

    for i in range(N):
        dist_sum = 0.0
        for j in range(N):
            if i == j:
                continue

            alignment = dtw(trajectories[i], trajectories[j], dist_method='euclidean')
            d = alignment.distance
            dist_sum += d

        m_values[i] = dist_sum / (N - 1)

    # 3. 计算统计量 mu_m 和 sigma_m
    mu_m = np.mean(m_values)
    sigma_m = np.std(m_values, ddof=1)  # ddof=1 对应 MATLAB 的 std 默认无偏估计

    # 防止除零错误
    if sigma_m < 1e-6:
        q_i_DTW = np.ones(N) / N
        return q_i_DTW

    # 4. 计算 Softmax 质量分数
    exponents = np.exp(-alpha_m * (m_values - mu_m) / sigma_m)
    q_i_DTW = exponents / np.sum(exponents)

    return q_i_DTW


# ================= 测试代码 =================
if __name__ == "__main__":
    # 1. 准备测试数据
    np.random.seed(42)  # 固定随机种子，保证每次运行结果一致
    demoNum = 5  # 演示数据数量
    demoLen = 50  # 每条数据的长度
    alpha_m = 1.0  # 缩放系数

    # 生成 4 条正常的正弦波轨迹（带一点随机噪声）
    t = np.linspace(0, 2 * np.pi, demoLen)
    normal_trajs = np.sin(t) + np.random.normal(0, 0.05, (4, demoLen))

    # 生成 1 条异常轨迹（形状差异较大）
    abnormal_traj = np.cos(t) * 2 + np.random.normal(0, 0.1, demoLen)

    # 拼接成 MATLAB 格式的宽矩阵 [1 x (demoNum * demoLen)]
    Datas = np.vstack((normal_trajs, abnormal_traj)).reshape(1, -1)

    # 2. 调用函数计算质量分数
    scores = calculate_q_DTW(Datas, demoNum, demoLen, alpha_m)

    # 3. 打印测试结果
    print("=" * 40)
    print("DTW 质量分数测试结果:")
    print("=" * 40)
    for i, score in enumerate(scores):
        label = "正常轨迹" if i < 4 else "异常轨迹"
        print(f"轨迹 {i + 1} ({label}): 质量分数 = {score:.4f}")

    print("\n结论: 异常轨迹的质量分数应该显著低于正常轨迹。")

