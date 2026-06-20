def calculate_parameter_count(config):
    arch = config['architecture']
    V = arch['vocab_size']
    H = arch['hidden_size']
    I = arch['intermediate_size']
    L = arch['num_layers']
    A = arch['num_attention_heads']
    KV = arch['num_key_value_heads']
    
    head_dim = H // A
    
    # 1. Embeddings
    embed_params = V * H
    
    # 2. Transformer Blocks
    # Attention: Q, K, V, and Output projections
    attn_params = (H * H) + 2 * (KV * head_dim * H) + (H * H)
    # MLP: Gate, Up, and Down projections
    mlp_params = 3 * I * H
    # RMSNorms: Attention norm and MLP norm per layer
    norm_params = 2 * H
    layer_params = L * (attn_params + mlp_params + norm_params)
    
    # 3. Final layer
    final_params = H + (V * H) # Final RMSNorm + LM Head
    
    return embed_params + layer_params + final_params

def calculate_benchmarks(runtime_info, config_dict):
    step_times = runtime_info['step_times']
    total_tokens = runtime_info['total_tokens_processed']
    
    # 1. Compile Time
    # JAX compiles on the first step. Compile time is the difference between step 1 and average step time.
    first_step = step_times[0] if step_times else 0
    avg_subsequent_step = sum(step_times[1:]) / len(step_times[1:]) if len(step_times) > 1 else first_step
    compile_time = max(0, first_step - avg_subsequent_step)
    
    # 2. Training Time & Step Time
    training_time = sum(step_times[1:]) if len(step_times) > 1 else 0
    
    # 3. Throughput
    tokens_per_sec = total_tokens / training_time if training_time > 0 else 0
    samples_per_sec = tokens_per_sec / config_dict['training']['sequence_length'] if training_time > 0 else 0
    
    # 4. Model Size
    param_count = calculate_parameter_count(config_dict)
    
    # 5. Theoretical FLOPs
    # standard estimator: 6 FLOPs per parameter per token (Forward + Backward)
    theoretical_flops = 6 * param_count * total_tokens
    
    # 6. Achieved TFLOPs/sec
    tflops_per_sec = (6 * param_count * tokens_per_sec) / (10**12) if training_time > 0 else 0
    
    # 7. Peak Memory Usage 
    # This represents the absolute minimum memory footprint (Float32 parameters). 
    # True peak memory including activations requires jax.profiler.
    param_memory_gb = (param_count * 4) / (1024**3) 
    
    return {
        "hardware_backend": config_dict['hardware']['backend'],
        "total_parameters": param_count,
        "theoretical_min_memory_gb": round(param_memory_gb, 4),
        "compile_time_seconds": round(compile_time, 4),
        "training_time_seconds": round(training_time, 4),
        "avg_step_time_seconds": round(avg_subsequent_step, 4),
        "throughput_tokens_per_sec": round(tokens_per_sec, 2),
        "throughput_samples_per_sec": round(samples_per_sec, 2),
        "theoretical_total_flops": theoretical_flops,
        "achieved_tflops_per_sec": round(tflops_per_sec, 6),
        "loss_trajectory": [round(l, 4) for l in runtime_info['loss_values']]
    }