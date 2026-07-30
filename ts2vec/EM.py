import numpy as np
from scipy.stats import multivariate_normal
class EM:
    def __init__(self,demoNum,demoLen,pdfNum,lambda_s=10.0,max_tier=100,tol=1e-6,D_z=1):

        self.D_z=D_z
        self.demoNum=demoNum
        self.demoLen=demoLen
        self.pdfNum=pdfNum
        self.max_tier=max_tier
        self.lambda_s=lambda_s
        self.tol=tol

    def initialize_para(self,Data_x,Data_f,S_it):
        #加载数据
        Data_x=Data_x.reshape(self.demoNum,self.demoLen)
        self.Data_x=Data_x[0]
        print(f"Data_x.shape:{Data_x.shape}")
        Data_f=Data_f.reshape(self.demoNum,self.demoLen)
        self.Data_f=Data_f
        print(f"Data_f.shape:{Data_f.shape}")
        self.S_it=S_it
        print(f"S_it.shape:{S_it.shape}")

        #初始化GMM参数
        self.Z = np.zeros((self.demoLen, self.D_z))
        self.pis = np.ones(self.pdfNum) / self.pdfNum
        self.mus = np.random.randn(self.pdfNum, self.D_z+1) * 0.1
        self.Sigmas = np.array([np.eye(self.D_z+1) for _ in range(self.pdfNum)])
        self.sigma_f_sq = 1.0
        print(f"Z.shape:{self.Z.shape}")
        print(f"pis")
    def E_step(self):
        # 1. 批量构建输入向量 (demoLen, D_z+1)
        # self.Data_x 是 (demoLen,) -> reshape 为 (demoLen, 1)
        # self.Z 是 (demoLen,Dz)
        X = np.concatenate([self.Data_x.reshape(-1, 1), self.Z], axis=1)
        print(f"X维度:{X.shape}")
        # 2. 向量化计算所有高斯分量的概率 (demoLen, pdfNum)
        weighted_probs = np.zeros((self.demoLen, self.pdfNum))
        for k in range(self.pdfNum):
            prob = multivariate_normal.pdf(X, mean=self.mus[k], cov=self.Sigmas[k])
            weighted_probs[:, k] = self.pis[k] * prob

        # 3. 归一化 (沿 axis=1 求和，保持维度为 (demoLen, 1) 以便广播除法)
        sum_probs = np.sum(weighted_probs, axis=1, keepdims=True) + 1e-6
        gamma = weighted_probs / sum_probs

        # 4. 转置为(pdfNum, demoLen)
        self.gamma=gamma.T
        print(f"gamma维度:{self.gamma.shape}")
        return gamma.T

    def M_step(self):
        # 计算Nk
        N = np.sum(self.gamma, axis=1)  # (K,)
        print(f"Nk维度:{N.shape}")
        # 计算pi_k
        sum_N = np.sum(N)
        pi = N / sum_N  # (K,)
        print(f"pi_k维度:{pi.shape}")
        # 计算mu_k
        X = np.concatenate([self.Data_x.reshape(-1, 1), self.Z], axis=1)  # (T, dim)
        mu = (self.gamma @ X).T / N  # (dim, K)
        print(f"mu_k维度:{mu.shape}")
        # 计算sigma_k
        K = self.gamma.shape[0]
        T = X.shape[0]
        dim = X.shape[1]

        Sigma = np.zeros((K, dim, dim))

        for k in range(K):
            diff = X - mu[:, k]  # (T, dim)
            weighted_diff = self.gamma[k, :, np.newaxis] * diff  # (T, dim)
            Sigma[k] = (weighted_diff.T @ diff) / N[k] + 1e-6 * np.eye(dim)  # (dim, dim)
        print(f"sigma_m维度:{Sigma.shape}")

        #计算sigma_f2
        if self.D_z == 1:
            # Data_f: (demoNum, demoLen)，z_t: (demoLen, 1)
            z_flat = self.Z.reshape(-1)  # (demoLen,)
            squared_errors = (self.Data_f - z_flat) ** 2  # (demoNum, demoLen)
        else:
            # 需要确保 Data_f 是 (demoNum, demoLen, D_z)
            # 如果 Data_f 是 (demoNum, demoLen)，需要 reshape
            if len(self.Data_f.shape) == 2:
                # 假设 D_z > 1 但 Data_f 是二维的，将其扩展到三维
                # 这种情况可能不对，需要根据实际情况调整
                raise ValueError(
                    f"Data_f shape {self.Data_f.shape} but D_z={self.D_z}. Expected (demoNum, demoLen, D_z)")

            # Data_f: (demoNum, demoLen, D_z)
            # Z: (demoLen, D_z) -> (1, demoLen, D_z) 用于广播
            diff = self.Data_f - self.Z[np.newaxis, :, :]  # (demoNum, demoLen, D_z)
            squared_errors = np.sum(diff ** 2, axis=2)  # (demoNum, demoLen)

        # 加权求和
        numerator = np.sum(self.S_it * squared_errors)
        denominator = np.sum(self.S_it)
        sigma_f_sq = numerator / (denominator + 1e-6)
        print(f"sigma_f_sq维度:{sigma_f_sq.shape}")
        #计算z
        self.N_k=N
        self.pis=pi
        self.mus=mu.T
        self.Sigmas=Sigma
        self.sigma_f_sq=sigma_f_sq
        self.update_z_analytic()
    def update_z_analytic(self):
        """
        解析法更新 z（D_z=1 时）
        求解一个三对角线性系统
        """
        T = self.demoLen
        D_z = self.D_z

        # 构建三对角线性系统 A * z = b

        # A 的对角线元素
        A_diag = np.zeros(T)
        A_offdiag = np.zeros(T - 1)  # 上/下对角线
        b = np.zeros(T)

        for t in range(T):
            # ----- 1. GMM 部分的贡献 -----
            # log N((x_t, z_t) | μ_k, Σ_k) 对 z_t 来说是二次的
            # 展开后：-1/2 * (z_t - μ_z|k)² / σ_z|k² + 常数
            # 其中 μ_z|k 是条件均值，σ_z|k² 是条件方差

            # 收集所有高斯分量的贡献
            A_gmm = 0.0
            b_gmm = 0.0

            for k in range(self.pdfNum):
                # 提取协方差矩阵的逆
                Sigma_inv = np.linalg.inv(self.Sigmas[k] + 1e-6 * np.eye(D_z + 1))  # (2, 2)

                # 提取 z 部分（第2行第2列，因为索引0是x，索引1是z）
                sigma_zz_inv = Sigma_inv[1, 1]  # 标量
                sigma_xz_inv = Sigma_inv[0, 1]  # 标量（x和z的交叉项）

                # 提取均值
                mu_x = self.mus[k, 0]  # x 的均值
                mu_z = self.mus[k, 1]  # z 的均值

                # 当前时间步的 x_t
                x_t = self.Data_x[t]

                # 贡献到 A 和 b
                # 二次项系数：γ * σ_zz_inv
                A_gmm += self.gamma[k, t] * sigma_zz_inv

                # 一次项系数：γ * [σ_zz_inv * μ_z - σ_xz_inv * (x_t - μ_x)]
                b_gmm += self.gamma[k, t] * (sigma_zz_inv * mu_z - sigma_xz_inv * (x_t - mu_x))

            # ----- 2. 数据拟合项的贡献 -----
            A_data = 0.0
            b_data = 0.0

            for n in range(self.demoNum):
                if self.S_it[n, t] > 0:
                    # (f_{n,t} - z_t)² / (2σ_f²)
                    # 展开：z_t²/(2σ_f²) - f_{n,t}*z_t/σ_f² + 常数
                    A_data += self.S_it[n, t] / self.sigma_f_sq
                    b_data += self.S_it[n, t] * self.Data_f[n, t] / self.sigma_f_sq

            # ----- 3. 组合 -----
            A_diag[t] = A_gmm + A_data
            b[t] = b_gmm + b_data

        # ----- 4. 添加平滑项（三对角线） -----
        # 平滑项：-(λ_s/2) * Σ_t (z_t - z_{t-1})²
        # 展开：-(λ_s/2) * Σ_t (z_t² - 2z_t*z_{t-1} + z_{t-1}²)
        # 对 z_t 求导：λ_s * (2z_t - z_{t-1} - z_{t+1})
        for t in range(T):
            if t > 0:
                A_diag[t] += self.lambda_s
                A_offdiag[t - 1] = -self.lambda_s  # 下对角线
            if t < T - 1:
                A_diag[t] += self.lambda_s
                # A_offdiag[t] = -self.lambda_s  # 上对角线（对称）

        # ----- 5. 求解三对角系统 -----
        # 使用 Thomas 算法（三对角矩阵求解器）
        from scipy.linalg import solve_banded

        # 构建带状矩阵
        # 下对角线、主对角线、上对角线
        ab = np.zeros((3, T))
        ab[0, 1:] = A_offdiag  # 下对角线
        ab[1, :] = A_diag  # 主对角线
        ab[2, :-1] = A_offdiag  # 上对角线

        # 求解
        z_new = solve_banded((1, 1), ab, b)

        self.Z = z_new.reshape(-1, 1)
        print(f"z维度:{self.Z.shape}")
        return self.Z

    def compute_log_likelihood(self):
        """计算当前参数下的对数似然（包括所有演示的数据项）"""
        X = np.concatenate([self.Data_x.reshape(-1, 1), self.Z], axis=1)
        log_lik = 0.0
        for t in range(self.demoLen):
            # 混合模型的似然：Σ_k π_k * N(x_t | μ_k, Σ_k)
            mix_prob = 0.0
            for k in range(self.pdfNum):
                prob = multivariate_normal.pdf(X[t], mean=self.mus[k], cov=self.Sigmas[k])
                mix_prob += self.pis[k] * prob
            log_lik += np.log(mix_prob + 1e-10)
        return log_lik





    #E,M步测试
if __name__ == "__main__":
    import pandas as pd
    from calculate_s_it import  calculate_s_it
    # 参数设置
    alpha_m = 0.5
    alpha_n = 0.5  # 缩放系数 (预留参数)
    alpha_c = 0.8
    lambda_c = 0.5
    D_z = 1
    demoNum = 10
    demoLen = 50
    pdfNum = 50
    lambda_s = 10.0
    max_iter = 100
    tol = 1e-6
    alpha_x = 1.0
    tau = 1.0
    x0 = 1.0  # t=0 时的初始值
    """
    # 生成假数据（实际使用时替换为真实数据）
    fake_x = np.random.randn(demoNum * demoLen)
    fake_f = np.random.randn(demoNum * demoLen)
    fake_s = np.random.rand(demoNum, demoLen)  # 权重建议非负
    """

    df = pd.read_csv('12345.txt', sep='\s+', header=None)
    raw_data = df.values.astype(np.float64)
    fake_x = x0 * np.exp(-(alpha_x / tau) * raw_data[0])
    fake_f=raw_data[2]
    fake_s=calculate_s_it(fake_f, demoNum, demoLen,alpha_m, alpha_n,alpha_c, lambda_c)

    print(fake_s)
    print("-------------------------------------")
    print(fake_x)
    print("-------------------------------------")
    print(fake_f)

    # 初始化模型
    model = EM(demoNum, demoLen, pdfNum, lambda_s=lambda_s, max_tier=max_iter, tol=tol, D_z=D_z)
    model.initialize_para(fake_x, fake_f, fake_s)

    # 训练循环
    prev_log_lik = -np.inf
    for iteration in range(max_iter):
        # E-step
        gamma = model.E_step()
        print(gamma)
        # M-step（内部会调用 update_z_analytic）
        model.M_step()

        # 计算当前对数似然
        curr_log_lik = model.compute_log_likelihood()
        print(f"Iteration {iteration+1}: log-likelihood = {curr_log_lik:.6f}")

        # 收敛检查
        if abs(curr_log_lik - prev_log_lik) < tol:
            print(f"Converged at iteration {iteration+1}")
            break
        prev_log_lik = curr_log_lik

    # 输出最终结果
    print("Training finished.")
    print(f"Final Z shape: {model.Z.shape}")
    print(f"Final mu shape: {model.mus.shape}")
    print(f"Final sigma_f_sq: {model.sigma_f_sq}")








