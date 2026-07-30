import numpy as np
import pandas as pd
import os

# 假设你的 load_forecast_csv 函数在 datautils.py 中
from datautils import load_forecast_csv


# ================= 1. 模拟生成一个符合要求的 CSV 文件 =================
def create_dummy_csv(file_name='test_dummy_data.csv'):
    # 生成 100 个时间步的连续小时数据
    dates = pd.date_range(start='2023-01-01', periods=100, freq='H')
    # 生成两个数值列
    data = {
        'date': dates,
        'sensor_1': np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100),
        'sensor_2': np.cos(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
    }
    df = pd.DataFrame(data)
    df.to_csv(file_name, index=False)
    print(f"[准备完毕] 成功生成模拟数据文件: {os.path.abspath(file_name)}")
    return file_name


# ================= 2. 测试数据加载 =================
def test_load_forecast_csv():
    # 1. 生成模拟数据
    csv_filename = create_dummy_csv()
    dataset_name = csv_filename.replace('.csv', '')  # 提取数据集名称，例如 'test_dummy_data'

    # 2. 关键修改：将当前工作目录切换到脚本所在目录
    # 这样，datautils.py 中的相对路径 'datasets/...' 就会以这里为起点
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"[路径已切换] 当前工作目录: {os.getcwd()}")

    # 3. 调用函数加载数据
    # 现在函数会尝试在 script_dir/datasets/test_dummy_data.csv 寻找文件
    # 但我们的文件在 script_dir/test_dummy_data.csv
    # 所以我们需要一个变通方法：创建一个临时的 datasets 软链接，或者直接修改函数。
    # 为了最小化侵入，我们采用一个更直接的方法：
    # 临时修改 datautils.py 中的路径逻辑是不现实的。
    # 最好的办法是，把生成的文件移动到它期望的 datasets 文件夹下。

    datasets_dir = os.path.join(script_dir, 'datasets')
    if not os.path.exists(datasets_dir):
        os.makedirs(datasets_dir)
        print(f"[目录创建] 创建 datasets 目录: {datasets_dir}")

    target_path = os.path.join(datasets_dir, csv_filename)
    os.replace(csv_filename, target_path)
    print(f"[文件已移动] {csv_filename} -> {target_path}")

    try:
        result = load_forecast_csv(dataset_name, univar=False)
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return

    # 4. 解包返回值
    data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = result

    # 5. 打印各项返回值的格式和形状
    print("\n" + "=" * 50)
    print("📊 返回值格式与形状检查:")
    print("=" * 50)

    print(f"1. data (核心数据):")
    print(f"   - 类型: {type(data)}")
    print(f"   - 形状: {data.shape}  (预期: [1, 100, 原始列数+7])")

    print(f"\n2. 数据集切片 (Slices):")
    print(f"   - train_slice: {train_slice} -> 训练集长度: {len(range(*train_slice.indices(data.shape[1])))}")
    print(f"   - valid_slice: {valid_slice} -> 验证集长度: {len(range(*valid_slice.indices(data.shape[1])))}")
    print(f"   - test_slice:  {test_slice} -> 测试集长度:  {len(range(*test_slice.indices(data.shape[1])))}")

    print(f"\n3. scaler (标准化器):")
    print(f"   - 类型: {type(scaler)}")
    print(f"   - 均值 (mean): {scaler.mean_[:3]}... (仅展示前3个)")

    print(f"\n4. pred_lens (预测长度列表):")
    print(f"   - 值: {pred_lens}")

    print(f"\n5. n_covariate_cols (时间特征数):")
    print(f"   - 值: {n_covariate_cols} (预期: 7)")

    print("=" * 50)
    print("✅ 测试完成！")


# 运行测试
if __name__ == "__main__":
    test_load_forecast_csv()