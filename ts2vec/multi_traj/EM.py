import numpy as np
from scipy.stats import multivariate_normal
from scipy.linalg import solve_banded


class EM:
    def __init__(self, K=10, lambda_s=10):
        self.K = K
        self.lambda_s = lambda_s
    """
    def loading_Data(self, X, F, S):
        #1. 取维度
        self.Num, self.T = F.shape

        #2. 加载数据
        self.X = X
        self.F = F
        self.S = S
        #3. 初始化执行EM算法需要的所有参数
        self.gamma = np.zeros((self.K,self.T))
        self.N = np.zeros(self.K)
        self.pi = np.ones(self.K) / self.K
        self.mean = np.zeros((2, self.K))
        self.sigma = np.zeros((self.K, 2, 2))
        for i in range(self.K):
            self.sigma[i] = np.eye(2)
        self.sigma_f2 = 1.0
        self.Z = np.mean(self.F, axis=0)
    """

    def loading_Data(self, X, F, S):
        # 1. 取维度
        self.Num, self.T = F.shape

        # 2. 加载数据
        self.X = X
        self.F = F
        self.S = S

        # 3. 初始化隐变量 Z (T,)
        # 用S加权的F均值作为Z的初始估计，比简单平均更合理
        weight_sum = np.sum(self.S, axis=0)  # (T,)
        self.Z = np.sum(self.S * self.F, axis=0) /  (weight_sum + 1e-10)
        print(self.Z.shape)
        # 4. 初始化 sigma_f2
        # 用Z初始值和F之间的加权方差来初始化
        diff_sq = (self.F - self.Z[np.newaxis, :]) ** 2  # (Num, T)
        numerator = np.sum(self.S * diff_sq)
        denominator = np.sum(self.S)
        self.sigma_f2 = max(numerator / (denominator + 1e-10), 1e-6)

        # 5. 用K-Means初始化GMM参数
        # 构造联合数据 (T, 2)
        X_Z = np.vstack((self.X, self.Z)).T  # (T, 2)

        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.K, n_init=10, random_state=42).fit(X_Z)

        # 均值: (2, K) —— 保持你原有的 (2, K) 布局
        self.mean = kmeans.cluster_centers_.T  # (2, K)

        # 协方差: 每个分量用簇内样本协方差
        self.sigma = np.zeros((self.K, 2, 2))
        labels = kmeans.labels_
        for k in range(self.K):
            mask = (labels == k)
            if np.sum(mask) > 2:
                cov = np.cov(X_Z[mask].T)
                self.sigma[k] = cov + 1e-4 * np.eye(2)
            else:
                # 样本不足时用全局协方差
                self.sigma[k] = np.cov(X_Z.T) + 1e-4 * np.eye(2)

        # 混合权重: 按簇大小比例
        counts = np.bincount(labels, minlength=self.K)
        self.pi = counts / counts.sum()
        self.pi = np.maximum(self.pi, 1e-6)
        self.pi /= self.pi.sum()

        # 6. 初始化gamma和N占位（E步会重新计算）
        self.gamma = np.ones((self.K, self.T)) / self.K
        self.N = np.ones(self.K) / self.K

        print(f"[Init] Z range: [{self.Z.min():.4f}, {self.Z.max():.4f}]")
        print(f"[Init] sigma_f2: {self.sigma_f2:.6f}")
        print(f"[Init] pi: {self.pi}")


    def E_step(self):
        #1. 构造联合向量[X,Z],并转置以便利用现有高斯函数轮子
        X_Z = np.vstack((self.X, self.Z))
        Data=X_Z.T

        #2. 将以上的数据计算正态分布概率
        weighted_pdf = np.zeros((self.K,self.T))
        for i in range(self.K):
            mean_k=self.mean[:,i] #2 x 1
            cov_k=self.sigma[i] #2 x 2
            weighted_pdf[i] = self.pi[i] * multivariate_normal.pdf(Data, mean=mean_k, cov=cov_k) #1 x T

        #3. 归一化处理
        sum_weighted_pdf = np.sum(weighted_pdf,axis=0)
        self.gamma = weighted_pdf / sum_weighted_pdf #K x T


    def M_step(self):
        #1. 计算N_k
        self.N = np.sum(self.gamma, axis=1)

        #2. 计算pi_k
        N_sum = np.sum(self.N)
        self.pi = self.N / N_sum

        #3. 计算mean_k
        X_Z = np.vstack((self.X, self.Z)) #2 x T
        self.mean = (self.gamma @ X_Z.T) / self.N[:, np.newaxis] # 形状 (K, 2)
        self.mean = self.mean.T #2 x K

        #4. 计算sigma_k
        for k in range(self.K):
            # 当前分量的均值向量 (2,)
            mu_k = self.mean[:, k]  # 2 X 1

            # 所有样本相对于 mu_k 的偏差：形状 (2, T)
            diff = X_Z - mu_k[:, np.newaxis]

            # 加权外积和： (2, T) @ (T, 2) = (2, 2)
            gamma_k = self.gamma[k, :]  # (T,)
            weighted_sum = diff @ (diff.T * gamma_k[:, np.newaxis])  # (2, T) @ (T, 2) = (2,2)

            # 除以 N_k（加极小值防除零）
            self.sigma[k] = weighted_sum / self.N[k]

            #加正则化项保证半正定，避免后续求逆失败
            self.sigma[k] += 1e-6 * np.eye(2)
            self.sigma[k] = np.diag(np.diag(self.sigma[k])) + 1e-6 * np.eye(2)

        #5. 计算sigma_f2
        # self.Z 为 (T,)，广播后与 self.F (N, T) 逐元素相减
        diff_sq = (self.F - self.Z) ** 2 #N x T
        numerator = np.sum(self.S * diff_sq) #N x T
        denominator = np.sum(self.S)
        self.sigma_f2 = numerator / denominator

        #6. 更新Z
        self.Z = self.update_z()
    """
    def update_z(self,):
        T = self.T
        K = self.K
        N = self.F.shape[0]
        if self.S is None:
            self.S = np.ones((N, T))  # 默认权重全1

        # 预先计算每个分量的协方差逆矩阵
        Q = np.linalg.inv(self.sigma)  # (K,2,2)
        q_zz = Q[:, 1, 1]  # (K,)
        q_xz = Q[:, 0, 1]  # (K,)

        diag = np.zeros(T)  # 主对角线
        offdiag = -self.lambda_s * np.ones(T - 1)  # 次对角线
        b = np.zeros(T)  # 右侧向量

        for t in range(T):
            # GMM 贡献
            gamma_t = self.gamma[:, t]  # (K,)
            mu_z = self.mean[1, :]  # (K,)  注意 self.mean 形状 (2,K)
            mu_x = self.mean[0, :]
            x_t = self.X[t]  # 标量

            diag[t] += np.sum(gamma_t * q_zz)
            b[t] += np.sum(gamma_t * (q_zz * mu_z - q_xz * (x_t - mu_x)))

            # 观测贡献
            s_t = self.S[:, t]  # (N,)
            f_t = self.F[:, t]  # (N,)
            sum_s = np.sum(s_t)
            sum_sf = np.sum(s_t * f_t)
            diag[t] += sum_s / self.sigma_f2
            b[t] += sum_sf / self.sigma_f2

        # 平滑项的边界和内部处理
        diag[0] += self.lambda_s
        diag[-1] += self.lambda_s
        for t in range(1, T - 1):
            diag[t] += 2 * self.lambda_s

        # 构造带状矩阵（下带宽=1，上带宽=1）
        ab = np.zeros((3, T))
        ab[1, :] = diag
        if T > 1:
            ab[0, 1:] = offdiag  # 上对角线
            ab[2, :-1] = offdiag  # 下对角线

        # 求解
        z_new = solve_banded((1, 1), ab, b)
        return z_new
    """

    def update_z(self):
        T = self.T
        K = self.K
        N = self.F.shape[0]

        # 1. 强制协方差对角化（关键！消除交叉项导致的震荡）
        # 如果 M_step 中还没做对角化，请在这里强制处理
        sigma_diag = np.zeros_like(self.sigma)
        for k in range(K):
            sigma_diag[k] = np.diag(np.diag(self.sigma[k])) + 1e-6 * np.eye(2)

        # 2. 计算逆矩阵（现在只有对角元素非零）
        Q = np.linalg.inv(sigma_diag)  # (K, 2, 2)
        q_zz = Q[:, 1, 1]  # (K,)
        # ★ 注意：对角化后 q_xz 理论上为 0，但为了公式完整性保留
        q_xz = Q[:, 0, 1]  # (K,)

        # 3. 初始化三对角矩阵（保证平滑项基底）
        diag = np.zeros(T)
        offdiag = -self.lambda_s * np.ones(T - 1)
        b = np.zeros(T)

        # 4. 遍历时间步构建方程组
        for t in range(T):
            gamma_t = self.gamma[:, t]  # (K,)
            mu_z = self.mean[1, :]  # (K,)
            mu_x = self.mean[0, :]  # (K,)
            x_t = self.X[t]  # 标量

            # ★ 核心修复：GMM 梯度项符号更正为 "+"
            gmm_pull = np.sum(gamma_t * (q_zz * mu_z + q_xz * (x_t - mu_x)))

            diag[t] += np.sum(gamma_t * q_zz)
            b[t] += gmm_pull

            # 观测项（带安全下限）
            s_t = self.S[:, t]
            f_t = self.F[:, t]
            sum_s = np.sum(s_t)
            sum_sf = np.sum(s_t * f_t)
            safe_sigma_f2 = max(self.sigma_f2, 1e-3)

            diag[t] += sum_s / safe_sigma_f2
            b[t] += sum_sf / safe_sigma_f2

        # 5. 添加平滑正则项
        diag[0] += self.lambda_s
        diag[-1] += self.lambda_s
        if T > 2:
            diag[1:-1] += 2.0 * self.lambda_s

        # 6. 求解带状方程组
        ab = np.zeros((3, T))
        ab[1, :] = diag
        if T > 1:
            ab[0, 1:] = offdiag
            ab[2, :-1] = offdiag

        z_new = solve_banded((1, 1), ab, b)
        return z_new




