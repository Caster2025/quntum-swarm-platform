import json
from qiskit.providers.aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error, readout_error

def build_ibm_lima_noise_model(config_path="./noise_model/IBM_Lima_noise.json"):
    """构建IBM Lima实测噪声模型"""
    with open(config_path, 'r') as f:
        noise_config = json.load(f)
    
    noise_model = NoiseModel()
    
    # 1. 热弛豫噪声
    t1 = noise_config["T1"]
    t2 = noise_config["T2"]
    thermal_error_1q = thermal_relaxation_error(t1, t2, 100e-9)  # 门操作时间100ns
    thermal_error_2q = thermal_relaxation_error(t1, t2, 500e-9)  # 两比特门操作时间500ns
    
    # 2. 退极化噪声
    depolar_1q = depolarizing_error(noise_config["single_qubit_depolar_prob"], 1)
    depolar_2q = depolarizing_error(noise_config["two_qubit_crosstalk_prob"], 2)
    
    # 3. 读取噪声
    ro_error = readout_error([[1 - noise_config["readout_error_0to1"], noise_config["readout_error_0to1"]],
                             [noise_config["readout_error_1to0"], 1 - noise_config["readout_error_1to0"]]])
    
    # 4. 添加到噪声模型
    noise_model.add_all_qubit_quantum_error(thermal_error_1q.compose(depolar_1q), ["u1", "u2", "u3"])
    noise_model.add_all_qubit_quantum_error(thermal_error_2q.compose(depolar_2q), ["cx"])
    noise_model.add_all_qubit_readout_error(ro_error)
    
    return noise_model
