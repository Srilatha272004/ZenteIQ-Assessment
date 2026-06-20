import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Tuple

class QwenConfig:
    vocab_size: int = 151936
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6

class RMSNorm(nn.Module):
    dim: int
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x):
        weight = self.param('weight', nn.initializers.ones, (self.dim,))
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        return (x / jnp.sqrt(variance + self.eps)) * weight

def apply_rotary_emb(x, freqs_cis):
    x_shape = x.shape
    x_reshaped = x.reshape(*x.shape[:-1], -1, 2)
    x_complex = x_reshaped[..., 0] + 1j * x_reshaped[..., 1]
    x_out = x_complex * freqs_cis
    x_out = jnp.stack([jnp.real(x_out), jnp.imag(x_out)], axis=-1)
    return x_out.reshape(*x_shape)

class QwenMLP(nn.Module):
    config: QwenConfig

    @nn.compact
    def __call__(self, x):
        gate_proj = nn.Dense(self.config.intermediate_size, use_bias=False)(x)
        up_proj = nn.Dense(self.config.intermediate_size, use_bias=False)(x)
        activated = nn.silu(gate_proj) * up_proj
        down_proj = nn.Dense(self.config.hidden_size, use_bias=False)(activated)
        return down_proj

class QwenAttention(nn.Module):
    config: QwenConfig

    @nn.compact
    def __call__(self, x, freqs_cis, mask=None):
        B, L, _ = x.shape
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        
        q_proj = nn.Dense(self.config.num_attention_heads * head_dim, use_bias=True)(x)
        k_proj = nn.Dense(self.config.num_key_value_heads * head_dim, use_bias=True)(x)
        v_proj = nn.Dense(self.config.num_key_value_heads * head_dim, use_bias=True)(x)

        q = q_proj.reshape(B, L, self.config.num_attention_heads, head_dim)
        k = k_proj.reshape(B, L, self.config.num_key_value_heads, head_dim)
        v = v_proj.reshape(B, L, self.config.num_key_value_heads, head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        num_rep = self.config.num_attention_heads // self.config.num_key_value_heads
        k = jnp.repeat(k, num_rep, axis=2)
        v = jnp.repeat(v, num_rep, axis=2)

        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        scores = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) / jnp.sqrt(head_dim)
        if mask is not None:
            scores = scores + mask

        probs = nn.softmax(scores, axis=-1)
        attn_out = jnp.matmul(probs, v)
        
        attn_out = jnp.transpose(attn_out, (0, 2, 1, 3)).reshape(B, L, -1)
        o_proj = nn.Dense(self.config.hidden_size, use_bias=False)(attn_out)
        return o_proj

class QwenBlock(nn.Module):
    config: QwenConfig

    @nn.compact
    def __call__(self, x, freqs_cis, mask=None):
        residual = x
        x = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)(x)
        x = QwenAttention(self.config)(x, freqs_cis, mask)
        x = x + residual

        residual = x
        x = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)(x)
        x = QwenMLP(self.config)(x)
        x = x + residual
        return x

class QwenModel(nn.Module):
    config: QwenConfig

    @nn.compact
    def __call__(self, input_ids):
        B, L = input_ids.shape
        embed_tokens = nn.Embed(self.config.vocab_size, self.config.hidden_size)
        x = embed_tokens(input_ids)

        head_dim = self.config.hidden_size // self.config.num_attention_heads
        freqs_cis = jnp.ones((B, L, 1, head_dim // 2), dtype=jnp.complex64) 
        
        mask = jnp.tril(jnp.ones((L, L)))
        mask = jnp.where(mask == 0, -1e9, 0.0)
        mask = mask.reshape(1, 1, L, L)

        for _ in range(self.config.num_layers):
            x = QwenBlock(self.config)(x, freqs_cis, mask)

        x = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)(x)
        logits = nn.Dense(self.config.vocab_size, use_bias=False)(x)
        return logits