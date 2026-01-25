# QSPAC：量子群体概率自适配计算框架
基于量子群体概率调控的硬件友好型量子计算框架，适配超导、离子阱、中性原子多硬件平台，在工业级噪声环境下实现高成功率、高资源利用率的量子计算任务执行。

## 仓库简介
本仓库包含QSPAC框架完整实现代码，支持**基准对比、多场景验证、多硬件适配、消融实验、极限负载测试、真实工业环境噪声实验**6大类核心实验，可直接复现论文核心结果。

## 仓库文件目录结构（务必核对，缺一不可）
quntum-swarm-platform/  # 仓库根目录
├── config/              # 配置文件目录
│   ├── noise_model/     # 噪声模型配置
│   │   └── IBM_Lima_noise.json  # IBM Lima实测噪声配置
│   ├── task_config/     # 任务配置
│   │   ├── H2_sto3g.json       # H2分子任务（核心基准任务）
│   │   ├── LiH_sto3g.json      # LiH分子任务（多场景验证）
│   │   └── max_cut_6node.json  # 6节点最大割任务（多场景验证）
│   └── framework_config/ # 框架配置
│       ├── QSPAC_config.json   # QSPAC框架核心配置
│       ├── Qiskit_config.json  # Qiskit对比框架配置
│       └── Cirq_config.json    # Cirq对比框架配置
├── experiment/          # 实验脚本目录
│   ├── noise_utils.py          # 噪声模型构建工具
│   ├── comparison_frameworks.py # Qiskit/Cirq对比框架封装
│   └── run_all_experiments.py  # 主实验脚本（所有实验入口）
├── data_analysis/       # 数据处理与可视化目录
│   └── analysis.py             # 结果分析+表格+图表生成
├── qspac_framework.py   # QSPAC框架核心实现（四大核心组件）
├── requirements.txt     # 依赖包清单（版本精准适配）
└── README.md            # 说明文档（当前文件）
## 快速复现步骤
### 前置准备
1.  电脑安装：Python 3.9.0（必须此版本，避免依赖冲突）、Git
2.  克隆本仓库到电脑：打开终端，输入命令
    ```bash
    git clone 你的仓库地址.git  # 仓库地址在你GitHub仓库主页绿色Code按钮复制
    cd quntum-swarm-platform   # 进入仓库目录
    ```

### 第一步：安装依赖（一键安装，无报错）
终端输入以下命令，自动安装所有所需包
```bash
pip install -r requirements.txt
第二步：运行所有实验（一键启动，无需额外配置）
 
终端输入命令，启动全部实验，结果自动保存python experiment/run_all_experiments.py

###第三步：生成实验结果表格与图表
 
终端输入命令，自动处理数据并生成可视化结果python experiment/run_all_experiments.py
运行完成后，仓库根目录会生成 result 文件夹，包含所有结果文件：
 
1. 原始数据（result/raw_data/）
 
- benchmark_results.json：基准对比实验原始数据
​
- multi_scene_results.json：多场景验证实验数据
​
- multi_hardware_results.json：多硬件适配实验数据
​
- ablation_results.json：消融实验数据
​
- 极限_load_results.json：极限负载测试数据
​
- real_industrial_noise_results.json：真实工业环境噪声实验数据
 
2. 对比表格（result/table/）
 
- benchmark_comparison.csv：基准对比实验表格（论文核心表）
​
- real_industrial_noise_comparison.csv：工业噪声实验对比表格（论文核心表）
 
3. 可视化图表（result/figure/）
 
- 极限_load_trend.png：极限负载测试成功率+资源利用率趋势图
​
- ablation_experiment.png：消融实验成功率+有效比特数柱状图
 
核心实验对应关系（覆盖论文所有验证）
 
1. 基准对比实验：H2分子任务+IBM Lima基础噪声，对比QSPAC/Qiskit/Cirq性能
​
2. 真实工业环境噪声实验：叠加比特非均匀性、时变漂移、关联噪声+突发干扰，工业场景验证
​
3. 极限负载测试：20比特硬件+10-300并行任务，验证框架过载适配能力
​
4. 消融实验：验证QSPAC四大核心组件（QPCC/NAGC/ORRC/LCSC）的必要性
​
5. 多场景验证：LiH分子+6节点最大割，验证框架多任务适配性
​
6. 多硬件适配：超导/离子阱/中性原子，验证跨硬件兼容性
 
关键注意事项
 
1. 运行环境必须是Python 3.9.0，版本过高/过低会导致依赖安装失败
​
2. 实验运行时间约10-15分钟（取决于电脑性能），耐心等待即可
​
3. 所有配置文件路径无需修改，代码已默认适配仓库目录结构
​
4. 手机端仅用于仓库文件管理，无法运行Python代码，必须在电脑端操作
​
5. 若运行报错，优先检查依赖包是否安装完整，或Python版本是否正确
