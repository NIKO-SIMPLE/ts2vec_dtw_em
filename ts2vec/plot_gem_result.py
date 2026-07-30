"""绘制 12345.txt 第三行的 10 条轨迹，以及 GEM 融合结果 z。"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 保存图片，不需要打开绘图窗口
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# 数据设置
N = 10       # 轨迹数量
T = 50       # 每条轨迹的时间步数
INPUT_ROW = 2  # 第三行；Python 的行号从 0 开始


def main():
    folder = Path(__file__).resolve().parent
    data_file = folder / "12345.txt"
    z_file = folder / "gem_z.txt"
    output_file = folder / "gem_trajectories_comparison.png"

    # 第三行的 500 个值按“第 1 条轨迹的 50 个时间步、...、第 10 条”排列。
    raw = np.loadtxt(data_file, dtype=np.float64)
    observations = raw[INPUT_ROW].reshape(N, T).T  # (500,) -> (T=50, N=10)

    # GEM 的一维 z：形状可以是 (50,) 或 (50, 1)。
    z = np.loadtxt(z_file, dtype=np.float64).reshape(-1)
    if z.size != T:
        raise ValueError(f"gem_z.txt 应有 {T} 个 z 值，实际读取到 {z.size} 个。")

    time = np.arange(1, T + 1)
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)

    # 10 条原始观测轨迹
    for i in range(N):
        ax.plot(
            time,
            observations[:, i],
            color="#3b82f6",
            alpha=0.38,
            linewidth=1.2,
            label="原始轨迹" if i == 0 else None,
        )

    # 最终融合轨迹 z
    ax.plot(time, z, color="#dc2626", linewidth=3, label="融合结果 z")

    ax.set_xlabel("时间步")
    ax.set_ylabel("数值")
    ax.set_title("10 条原始轨迹与 GEM 融合轨迹")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.savefig(output_file, dpi=180)
    plt.close(fig)
    print(f"图片已保存到: {output_file}")


if __name__ == "__main__":
    main()
