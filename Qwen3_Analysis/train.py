import os
import yaml

# 1. Universal Hardware Router (Must run before JAX is imported)
with open("/content/drive/MyDrive/Qwen0.6B_AnalysisRealdata/configs/qwen0.8tpu.yml", "r") as f:
    _pre_config = yaml.safe_load(f)
_backend = _pre_config.get('hardware', {}).get('backend', 'cpu')

if _backend == 'gpu':
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.95'
elif _backend == 'tpu':
    try:
        import jax.tools.colab_tpu
        jax.tools.colab_tpu.setup_tpu()
    except Exception as e:
        print(f"TPU Setup Failed (Are you on a TPU runtime?): {e}")

import time
import json
import jax
import jax.numpy as jnp
import numpy as np
import optax
import grain.python as grain
from array_record.python import array_record_module
from flax.jax_utils import replicate, unreplicate
from model import QwenModel, QwenConfig
import metrics

def load_config(config_path="/content/drive/MyDrive/Qwen0.6B_AnalysisRealdata/configs/qwen0.8tpu.yml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def setup_environment():
    num_devices = jax.device_count()
    device_type = jax.devices()[0].platform.upper()
    print(f"Executing on {num_devices} {device_type} core(s).")
    return num_devices

def generate_array_record_data(config, record_path="dataset.array_record"):
    seq_length = config['training']['sequence_length']
    vocab_size = config['architecture']['vocab_size']
    num_samples = config['dataset']['num_samples']
    
    print(f"Writing {num_samples} binary records to {record_path}...")
    writer = array_record_module.ArrayRecordWriter(record_path, 'group_size:1')
    
    for _ in range(num_samples):
        sample = np.random.randint(0, vocab_size, size=(seq_length,), dtype=np.int32)
        writer.write(sample.tobytes())
    writer.close()
    return record_path

def create_grain_iterator(record_path, batch_size):
    print(f"Loading dataset with Grain...")
    data_source = grain.ArrayRecordDataSource(record_path)
    dataset = grain.MapDataset.source(data_source)
    
    dataset = dataset.map(lambda x: np.frombuffer(x, dtype=np.int32))
    dataset = dataset.repeat() 
    dataset = dataset.batch(batch_size, drop_remainder=True)
    
    return iter(dataset.to_iter_dataset())

def create_synthetic_iterator(batch_size, seq_length, vocab_size):
    print("Using synthetic dataset generator...")
    while True:
        yield jax.random.randint(jax.random.PRNGKey(int(time.time())), (batch_size, seq_length), 0, vocab_size)

def cross_entropy_loss(logits, labels):
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    loss = optax.softmax_cross_entropy_with_integer_labels(logits=shifted_logits, labels=shifted_labels)
    return jnp.mean(loss)

def main():
    config_dict = load_config()
    train_cfg = config_dict['training']
    arch_cfg = config_dict['architecture']
    
    num_devices = setup_environment()
    
    # Validation for TPU batch sizes
    if num_devices > 1 and train_cfg['batch_size'] % num_devices != 0:
        raise ValueError(f"Batch size ({train_cfg['batch_size']}) must be a multiple of device count ({num_devices}) for parallel execution.")
    
    dataset_type = config_dict['dataset']['type']
    
    if dataset_type == 'array_record':
        record_path = generate_array_record_data(config_dict)
        data_iterator = create_grain_iterator(record_path, train_cfg['batch_size'])
        
    elif dataset_type == 'synthetic':
        data_iterator = create_synthetic_iterator(
            train_cfg['batch_size'], 
            train_cfg['sequence_length'], 
            arch_cfg['vocab_size']
        )
    else:
        raise ValueError(f"Unknown dataset type specified in YAML: {dataset_type}")
    
    print("Initializing model and optimizer...")
    qwen_config = QwenConfig()
    for k, v in arch_cfg.items():
        setattr(qwen_config, k, v)
        
    model = QwenModel(qwen_config)
    rng = jax.random.PRNGKey(train_cfg['seed'])
    
    dummy_input = jnp.ones((train_cfg['batch_size'], train_cfg['sequence_length']), dtype=jnp.int32)
    params = model.init(rng, dummy_input)
    
    optimizer = optax.adamw(
        learning_rate=train_cfg['learning_rate'], 
        weight_decay=train_cfg['weight_decay']
    )
    opt_state = optimizer.init(params)
    
    # 2. Dynamic Compilation Switch (JIT for single device, PMAP for multi-device)
    if num_devices > 1:
        print("Replicating model weights across multiple cores...")
        params = replicate(params)
        opt_state = replicate(opt_state)
        
        @jax.pmap(axis_name='batch')
        def train_step(params, opt_state, batch):
            def loss_fn(p):
                logits = model.apply(p, batch)
                return cross_entropy_loss(logits.astype(jnp.float32), batch)
            
            loss, grads = jax.value_and_grad(loss_fn)(params)
            grads = jax.lax.pmean(grads, axis_name='batch')
            loss = jax.lax.pmean(loss, axis_name='batch')
            
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, loss
    else:
        @jax.jit
        def train_step(params, opt_state, batch):
            def loss_fn(p):
                logits = model.apply(p, batch)
                return cross_entropy_loss(logits.astype(jnp.float32), batch)
            
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, loss

    runtime_info = {
        "start_time": time.time(),
        "step_times": [],
        "loss_values": [],
        "total_tokens_processed": 0
    }
    
    print("Starting training loop...")
    for step, batch in enumerate(data_iterator):
        if step >= train_cfg['steps']:
            break
            
        step_start = time.time()
        
        # 3. Dynamic Batch Reshaping for PMAP
        if num_devices > 1:
            batch = batch.reshape((num_devices, batch.shape[0] // num_devices, batch.shape[1]))
            
        params, opt_state, loss = train_step(params, opt_state, batch)
        
        # Ensure we read the scalar loss correctly whether replicated or not
        loss_val = unreplicate(loss).item() if num_devices > 1 else loss.item()
        
        step_end = time.time()
        
        runtime_info["step_times"].append(step_end - step_start)
        runtime_info["loss_values"].append(loss_val)
        runtime_info["total_tokens_processed"] += (train_cfg['batch_size'] * train_cfg['sequence_length'])
        
        print(f"Step {step+1}/{train_cfg['steps']} | Loss: {loss_val:.4f} | Time: {step_end - step_start:.4f}s")
        
    runtime_info["end_time"] = time.time()
    
    final_metrics = metrics.calculate_benchmarks(runtime_info, config_dict)
    
    output_dir = config_dict['output']['base_output_directory']
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{config_dict['output']['run_name']}_metrics.json")
    
    with open(output_file, "w") as f:
        json.dump(final_metrics, f, indent=4)
        
    print(f"Execution complete. Metrics successfully saved to {output_file}")

if __name__ == "__main__":
    main()