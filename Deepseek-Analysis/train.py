import os
import yaml

# 1. Universal Hardware Router
with open("/content/drive/MyDrive/Deepseek_Analysis/configs/deepseek0.9bcpu.yml", "r") as f:
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
        print(f"TPU Setup Failed: {e}")

import time
import json
import jax
import jax.numpy as jnp
import numpy as np
import optax
import grain.python as grain
from array_record.python import array_record_module
from flax.jax_utils import replicate, unreplicate

# Import both the model and its explicit configuration dataclass
from model import DeepSeekMoEModel, MoEConfig
import metrics

def load_config(config_path="/content/drive/MyDrive/Deepseek_Analysis/configs/deepseek0.9bcpu.yml"):
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
    writer = array_record_module.ArrayRecordWriter(record_path, 'group_size:1')
    for _ in range(num_samples):
        sample = np.random.randint(0, vocab_size, size=(seq_length,), dtype=np.int32)
        writer.write(sample.tobytes())
    writer.close()
    return record_path

def create_grain_iterator(record_path, batch_size):
    data_source = grain.ArrayRecordDataSource(record_path)
    dataset = grain.MapDataset.source(data_source)
    dataset = dataset.map(lambda x: np.frombuffer(x, dtype=np.int32))
    dataset = dataset.repeat().batch(batch_size, drop_remainder=True)
    return iter(dataset.to_iter_dataset())

def create_synthetic_iterator(batch_size, seq_length, vocab_size):
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
    
    num_devices = setup_environment()
    
    if num_devices > 1 and train_cfg['batch_size'] % num_devices != 0:
        raise ValueError("Batch size must be a multiple of device count for PMAP.")
    
    dataset_type = config_dict['dataset']['type']
    if dataset_type == 'array_record':
        record_path = generate_array_record_data(config_dict)
        data_iterator = create_grain_iterator(record_path, train_cfg['batch_size'])
    elif dataset_type == 'synthetic':
        data_iterator = create_synthetic_iterator(train_cfg['batch_size'], train_cfg['sequence_length'], config_dict['architecture']['vocab_size'])
    else:
        raise ValueError("Unknown dataset type.")
    
    # Instantiate configuration via the formal dataclass to bypass FrozenDict conversion
    moe_config = MoEConfig(**config_dict['architecture'])
    model = DeepSeekMoEModel(moe_config)
    rng = jax.random.PRNGKey(train_cfg['seed'])
    
    dummy_input = jnp.ones((train_cfg['batch_size'], train_cfg['sequence_length']), dtype=jnp.int32)
    
    # Initialize variables to capture both parameters and intermediate layer metrics
    variables = model.init(rng, dummy_input)
    params = variables['params']
    
    optimizer = optax.adamw(learning_rate=train_cfg['learning_rate'], weight_decay=train_cfg['weight_decay'])
    opt_state = optimizer.init(params)
    
    routing_loss_weight = moe_config.routing_loss_weight

    def core_train_step(params, opt_state, batch):
        def loss_fn(p):
            logits, state = model.apply({'params': p}, batch, mutable=['intermediates'])
            task_loss = cross_entropy_loss(logits.astype(jnp.float32), batch)
            
            # Flax nests sown variables by layer name. Since load_balance_loss is the 
            # ONLY thing we sow into 'intermediates', we safely extract all leaves.
            intermediates_tree = state.get('intermediates', {})
            routing_losses = jax.tree_util.tree_leaves(intermediates_tree)
            
            # Sum them up. If the tree is empty (e.g., a bug), default to 0.0
            total_routing_loss = jnp.sum(jnp.array(routing_losses)) if routing_losses else 0.0
            
            total_loss = task_loss + (routing_loss_weight * total_routing_loss)
            return total_loss, (task_loss, total_routing_loss)
        
        (total_loss, (task_loss, routing_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        
        if num_devices > 1:
            grads = jax.lax.pmean(grads, axis_name='batch')
            total_loss = jax.lax.pmean(total_loss, axis_name='batch')
            task_loss = jax.lax.pmean(task_loss, axis_name='batch')
            routing_loss = jax.lax.pmean(routing_loss, axis_name='batch')
            
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, total_loss, task_loss, routing_loss

    if num_devices > 1:
        params = replicate(params)
        opt_state = replicate(opt_state)
        train_step = jax.pmap(core_train_step, axis_name='batch')
    else:
        train_step = jax.jit(core_train_step)

    runtime_info = {
        "start_time": time.time(),
        "step_times": [],
        "task_loss": [],
        "routing_loss": [],
        "total_loss": [],
        "total_tokens_processed": 0
    }
    
    print("Starting MoE training loop...")
    for step, batch in enumerate(data_iterator):
        if step >= train_cfg['steps']: break
            
        step_start = time.time()
        
        if num_devices > 1:
            batch = batch.reshape((num_devices, batch.shape[0] // num_devices, batch.shape[1]))
            
        params, opt_state, total_loss, task_loss, routing_loss = train_step(params, opt_state, batch)
        
        t_loss_val = unreplicate(task_loss).item() if num_devices > 1 else task_loss.item()
        r_loss_val = unreplicate(routing_loss).item() if num_devices > 1 else routing_loss.item()
        tot_loss_val = unreplicate(total_loss).item() if num_devices > 1 else total_loss.item()
        
        step_time = time.time() - step_start
        
        runtime_info["step_times"].append(step_time)
        runtime_info["task_loss"].append(t_loss_val)
        runtime_info["routing_loss"].append(r_loss_val)
        runtime_info["total_loss"].append(tot_loss_val)
        runtime_info["total_tokens_processed"] += (train_cfg['batch_size'] * train_cfg['sequence_length'])
        
        print(f"Step {step+1} | Total Loss: {tot_loss_val:.4f} (Task: {t_loss_val:.4f}, Balance: {r_loss_val:.4f}) | Time: {step_time:.4f}s")
        
    runtime_info["end_time"] = time.time()
    
    final_metrics = metrics.calculate_benchmarks(runtime_info, config_dict)
    
    output_dir = config_dict['output']['base_output_directory']
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"{config_dict['output']['run_name']}_metrics.json")
    
    with open(out_file, "w") as f:
        json.dump(final_metrics, f, indent=4)
        
    print(f"Execution complete. Saved to {out_file}")

if __name__ == "__main__":
    main()