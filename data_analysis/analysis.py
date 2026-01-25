import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体，解决图表中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_raw_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def generate_benchmark_table():
    """生成基准对比实验表格（对应原文表1）"""
    data = load_raw_data("../result/raw_data/benchmark_results.json")
    frameworks = ["QSPAC", "QiskitAer", "Cirq"]
    metrics = ["effective_qubits", "success_rate", "energy_error", "resource_utilization", 
               "crosstalk_loss", "gate_count", "schedule_delay"]
    
    # 构建表格数据
    table_data = []
    for metric in metrics:
        row = {"评估维度": metric}
        for fw in frameworks:
            mean = data[fw][metric]["mean"]
            std = data[fw][metric]["std"]
            row[fw] = f"{mean:.2f}±{std:.2f}"
        # 计算QSPAC相对优势
        qiskit_val = data["QiskitAer"][metric]["mean"]
        cirq_val = data["Cirq"][metric]["mean"]
        qspac_val = data["QSPAC"][metric]["mean"]
        if metric in ["effective_qubits", "energy_error", "crosstalk_loss", "gate_count", "schedule_delay"]:
            # 越小越优：节省比例
            qiskit_adv = (qiskit_val - qspac_val) / qiskit_val * 100
            cirq_adv = (cirq_val - qspac_val) / cirq_val * 100
            row["QSPAC相对优势"] = f"节省{qiskit_adv:.1f}%/{cirq_adv:.1f}%"
        else:
            # 越大越优：提升倍数
            qiskit_adv = qspac_val / qiskit_val
            cirq_adv = qspac_val / cirq_val
            row["QSPAC相对优势"] = f"提升{qiskit_adv:.1f}倍/{cirq_adv:.1f}倍"
        table_data.append(row)
    
    # 保存表格
    df = pd.DataFrame(table_data)
    df.to_csv("../result/table/benchmark_comparison.csv", index=False, encoding="utf-8-sig")
    print("基准对比表格已生成")
    return df

def generate_极限_load_plot():
    """生成极限负载测试可视化图表"""
    data = load_raw_data("../result/raw_data/极限_load_results.json")
    df = pd.DataFrame(data)
    
    # 绘制成功率和资源利用率趋势
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(df["task_count"], df["success_rate"] * 100, marker='o', label="计算成功率")
    ax1.axhline(y=99, color='r', linestyle='--', label="合格线(99%)")
    ax1.axvline(x=262, color='orange', linestyle='--', label="极限阈值(262个任务)")
    ax1.set_xlabel("并行任务数")
    ax1.set_ylabel("成功率(%)")
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(df["task_count"], df["resource_utilization"], marker='s', label="资源利用率")
    ax2.set_xlabel("并行任务数")
    ax2.set_ylabel("资源利用率(%)")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig("../result/figure/极限_load_trend.png", dpi=300, bbox_inches='tight')
    print("极限负载测试图表已生成")

def generate_ablation_plot():
    """生成消融实验可视化图表"""
    data = load_raw_data("../result/raw_data/ablation_results.json")
    cases = ["full", "without_QPCC", "without_NAGC", "without_ORRC", "without_LCSC"]
    case_names = ["完整框架", "移除QPCC", "移除NAGC", "移除ORRC", "移除LCSC"]
    
    # 提取核心指标
    success_rates = [np.mean([r["success_rate"] for r in data[case]]) * 100 for case in cases]
    effective_qubits = [np.mean([r["effective_qubits"] for r in data[case]]) for case in cases]
    
    # 绘制柱状图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar(case_names, success_rates, color=['green', 'red', 'orange', 'blue', 'purple'])
    ax1.set_ylabel("计算成功率(%)")
    ax1.set_ylim(0, 100)
    ax1.grid(True, axis='y')
    
    ax2.bar(case_names, effective_qubits, color=['green', 'red', 'orange', 'blue', 'purple'])
    ax2.set_ylabel("有效比特数")
    ax2.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig("../result/figure/ablation_experiment.png", dpi=300, bbox_inches='tight')
    print("消融实验图表已生成")

def generate_real_industrial_noise_table():
    """生成真实工业环境噪声实验表格（对应原文表3）"""
    data = load_raw_data("../result/raw_data/real_industrial_noise_results.json")
    frameworks = ["QSPAC", "QiskitAer", "Cirq"]
    metrics = ["effective_qubits", "success_rate", "energy_error", "resource_utilization", 
               "total_time", "crosstalk_loss", "gate_count", "schedule_delay"]
    
    table_data = []
    for metric in metrics:
        row = {"评估维度": metric}
        for fw in frameworks:
            mean = np.mean([r[metric] for r in data[fw]])
            std = np.std([r[metric] for r in data[fw]])
            if metric == "success_rate":
                row[fw] = f"{mean*100:.1f}%±{std*100:.1f}%"
            else:
                row[fw] = f"{mean:.3f}±{std:.3f}"
        # 计算QSPAC相对优势（和原文加权提升逻辑一致）
        qiskit_val = np.mean([r[metric] for r in data["QiskitAer"]])
        cirq_val = np.mean([r[metric] for r in data["Cirq"]])
        qspac_val = np.mean([r[metric] for r in data["QSPAC"]])
        if metric in ["effective_qubits", "energy_error", "total_time", "crosstalk_loss", "gate_count", "schedule_delay"]:
            qiskit_adv = (qiskit_val - qspac_val) / qiskit_val * 100
            cirq_adv = (cirq_val - qspac_val) / cirq_val * 100
            row["QSPAC相对优势"] = f"节省{qiskit_adv:.1f}%/{cirq_adv:.1f}%"
        else:
            qiskit_adv = qspac_val / qiskit_val
            cirq_adv = qspac_val / cirq_val
            row["QSPAC相对优势"] = f"提升{qiskit_adv:.1f}倍/{cirq_adv:.1f}倍"
        table_data.append(row)
    
    df = pd.DataFrame(table_data)
    df.to_csv("../result/table/real_industrial_noise_comparison.csv", index=False, encoding="utf-8-sig")
    print("真实工业环境噪声实验表格已生成")
    return df

def main():
    """生成所有分析结果和可视化图表"""
    generate_benchmark_table()
    generate_极限_load_plot()
    generate_ablation_plot()
    generate_real_industrial_noise_table()  # 调用新增工业噪声实验表格生成
    
    # 生成加权总提升百分比（对应原文5.2.2）
    benchmark_data = load_raw_data("../result/raw_data/benchmark_results.json")
    weights = {
        "effective_qubits": 0.3,
        "success_rate": 0.3,
        "energy_error": 0.15,
        "resource_utilization": 0.1,
        "total_time": 0.05,
        "crosstalk_loss": 0.05,
        "gate_count": 0.03,
        "schedule_delay": 0.02
    }
    
    # 计算相对于Qiskit Aer的加权提升
    qiskit_weights = 0
    for metric, weight in weights.items():
        qspac_val = benchmark_data["QSPAC"][metric]["mean"] if metric in benchmark_data["QSPAC"] else 0
        qiskit_val = benchmark_data["QiskitAer"][metric]["mean"] if metric in benchmark_data["QiskitAer"] else 1
        if metric in ["effective_qubits", "energy_error", "crosstalk_loss", "gate_count", "schedule_delay", "total_time"]:
            # 越小越优：提升率=(传统值-QSPAC值)/传统值
            gain = (qiskit_val - qspac_val) / qiskit_val if qiskit_val !=0 else 0
        else:
            # 越大越优：提升率=(QSPAC值-传统值)/传统值
            gain = (qspac_val - qiskit_val) / qiskit_val if qiskit_val !=0 else 0
        qiskit_weights += gain * weight
    
    print(f"相对于Qiskit Aer的加权总提升：{qiskit_weights * 100:.2f}%")
    
    # 计算相对于Cirq的加权提升
    cirq_weights = 0
    for metric, weight in weights.items():
        qspac_val = benchmark_data["QSPAC"][metric]["mean"] if metric in benchmark_data["QSPAC"] else 0
        cirq_val = benchmark_data["Cirq"][metric]["mean"] if metric in benchmark_data["Cirq"] else 1
        if metric in ["effective_qubits", "energy_error", "crosstalk_loss", "gate_count", "schedule_delay", "total_time"]:
            gain = (cirq_val - qspac_val) / cirq_val if cirq_val !=0 else 0
        else:
            gain = (qspac_val - cirq_val) / cirq_val if cirq_val !=0 else 0
        cirq_weights += gain * weight
    
    print(f"相对于Cirq的加权总提升：{cirq_weights * 100:.2f}%")
    print("所有分析完成！表格和图表已保存至result目录")

if __name__ == "__main__":
    main()
