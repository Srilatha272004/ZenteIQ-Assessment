import numpy as np

def calculate_benchmarks(runtime_info, config_dict):
    """
    Calculates step throughput and formats dual-loss MoE metrics.
    Deep hardware metrics (MFU, TFLOPs) are captured by the XProf TensorBoard plugin.
    """
    
    # 1. Remove the first step (compilation step) from averages
    if len(runtime_info["step_times"]) > 1:
        pure_step_times = runtime_info["step_times"][1:]
    else:
        pure_step_times = runtime_info["step_times"]
        
    avg_step_time = np.mean(pure_step_times)
    
    # 2. Calculate Throughput
    batch_size = config_dict['training']['batch_size']
    seq_len = config_dict['training']['sequence_length']
    tokens_per_step = batch_size * seq_len
    tokens_per_second = tokens_per_step / avg_step_time if avg_step_time > 0 else 0

    # 3. Format final payload
    metrics_report = {
        "run_name": config_dict['output']['run_name'],
        "hardware_backend": config_dict['hardware']['backend'],
        "model_architecture": "DeepSeek-MoE",
        "throughput_metrics": {
            "average_step_time_seconds": round(float(avg_step_time), 4),
            "tokens_per_second": round(float(tokens_per_second), 2),
            "total_tokens_processed": int(runtime_info["total_tokens_processed"]),
            "total_execution_time_seconds": round(float(runtime_info["end_time"] - runtime_info["start_time"]), 2)
        },
        "loss_curves": {
            "final_total_loss": round(float(runtime_info["total_loss"][-1]), 4),
            "final_task_loss": round(float(runtime_info["task_loss"][-1]), 4),
            "final_routing_loss": round(float(runtime_info["routing_loss"][-1]), 4),
            "total_loss_history": [round(float(x), 4) for x in runtime_info["total_loss"]],
            "task_loss_history": [round(float(x), 4) for x in runtime_info["task_loss"]],
            "routing_loss_history": [round(float(x), 4) for x in runtime_info["routing_loss"]]
        },
        "configuration_snapshot": {
            "num_experts": config_dict['architecture']['num_experts'],
            "top_k": config_dict['architecture']['top_k'],
            "batch_size": batch_size,
            "sequence_length": seq_len
        }
    }
    
    return metrics_report