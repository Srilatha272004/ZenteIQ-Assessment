import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any
from dataclasses import dataclass

# 1. DEFINE THE CONFIGURATION DATACLASS HERE SO IT CAN BE IMPORTED BY TRAIN.PY
@dataclass
class MoEConfig:
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    rms_norm_eps: float
    num_experts: int
    num_shared_experts: int
    expert_intermediate_size: int
    top_k: int
    routing_loss_weight: float

class RMSNorm(nn.Module):
    epsilon: float = 1e-6
    @nn.compact
    def __call__(self, x):
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        normed = x * jax.lax.rsqrt(variance + self.epsilon)
        return normed * self.param('scale', nn.initializers.ones, (x.shape[-1],))

class Attention(nn.Module):
    config: MoEConfig  # Updated type
    @nn.compact
    def __call__(self, x):
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        q = nn.Dense(self.config.hidden_size, use_bias=False)(x)
        k = nn.Dense(self.config.hidden_size, use_bias=False)(x)
        v = nn.Dense(self.config.hidden_size, use_bias=False)(x)
        
        q = q.reshape(x.shape[0], x.shape[1], self.config.num_attention_heads, head_dim)
        k = k.reshape(x.shape[0], x.shape[1], self.config.num_attention_heads, head_dim)
        v = v.reshape(x.shape[0], x.shape[1], self.config.num_attention_heads, head_dim)
        
        scores = jnp.einsum('bqhd,bkhd->bhqk', q, k) / jnp.sqrt(head_dim)
        attn = jax.nn.softmax(scores, axis=-1)
        out = jnp.einsum('bhqk,bkhd->bqhd', attn, v)
        out = out.reshape(x.shape[0], x.shape[1], self.config.hidden_size)
        return nn.Dense(self.config.hidden_size, use_bias=False)(out)

class Expert(nn.Module):
    hidden_size: int
    intermediate_size: int
    @nn.compact
    def __call__(self, x):
        gate = nn.Dense(self.intermediate_size, use_bias=False)(x)
        up = nn.Dense(self.intermediate_size, use_bias=False)(x)
        down = nn.Dense(self.hidden_size, use_bias=False)(nn.silu(gate) * up)
        return down

class DeepSeekMoELayer(nn.Module):
    config: MoEConfig  # Updated type
    @nn.compact
    def __call__(self, x):
        num_shared = self.config.num_shared_experts
        num_routed = self.config.num_experts - num_shared
        top_k = self.config.top_k
        
        # 1. Shared Experts (Always Active)
        shared_out = jnp.zeros_like(x)
        if num_shared > 0:
            for i in range(num_shared):
                shared_out += Expert(self.config.hidden_size, self.config.expert_intermediate_size, name=f"shared_{i}")(x)
                
        # 2. Routed Experts (Top-K Gating)
        gate_logits = nn.Dense(num_routed, use_bias=False, name="router")(x)
        routing_weights = jax.nn.softmax(gate_logits, axis=-1)
        top_k_weights, top_k_indices = jax.lax.top_k(routing_weights, top_k)
        
        # Normalize top-K probabilities
        top_k_weights = top_k_weights / (jnp.sum(top_k_weights, axis=-1, keepdims=True) + 1e-9)
        
        # Evaluate all routed experts and mask out the unused ones
        expert_outputs = [Expert(self.config.hidden_size, self.config.expert_intermediate_size, name=f"routed_{i}")(x) for i in range(num_routed)]
        expert_outputs = jnp.stack(expert_outputs, axis=-2)
        
        one_hot_indices = jax.nn.one_hot(top_k_indices, num_routed)
        weighted_mask = one_hot_indices * jnp.expand_dims(top_k_weights, -1)
        combined_weights = jnp.sum(weighted_mask, axis=2)
        
        routed_out = jnp.sum(expert_outputs * jnp.expand_dims(combined_weights, -1), axis=2)
        
        # 3. Load Balancing Loss Calculation
        expert_fractions = jnp.mean(jnp.sum(one_hot_indices, axis=2), axis=(0, 1))
        routing_probs = jnp.mean(routing_weights, axis=(0, 1))
        load_balance_loss = num_routed * jnp.sum(expert_fractions * routing_probs)
        
        # Store loss in JAX variable collection
        self.sow('intermediates', 'load_balance_loss', load_balance_loss)
        
        return shared_out + routed_out

class TransformerBlock(nn.Module):
    config: MoEConfig  # Updated type
    @nn.compact
    def __call__(self, x):
        h = x + Attention(self.config)(RMSNorm(self.config.rms_norm_eps)(x))
        h = h + DeepSeekMoELayer(self.config)(RMSNorm(self.config.rms_norm_eps)(h))
        return h

class DeepSeekMoEModel(nn.Module):
    config: MoEConfig  # Updated type
    @nn.compact
    def __call__(self, x):
        x = nn.Embed(self.config.vocab_size, self.config.hidden_size)(x)
        for i in range(self.config.num_layers):
            x = TransformerBlock(self.config, name=f"layer_{i}")(x)
        x = RMSNorm(self.config.rms_norm_eps)(x)
        logits = nn.Dense(self.config.vocab_size, use_bias=False)(x)
        return logits