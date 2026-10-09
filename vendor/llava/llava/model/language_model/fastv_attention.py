"""Independent FastV scoring; never requests decoder attention outputs.

Only the last valid prompt query is scored against all prompt keys. Projection
and RoPE use the model dtype; the small QK row and softmax use FP32 to avoid
FP16 overflow. This side calculation does not update hidden states or KV caches.
"""

import math

import torch


def _rotate_half(values):
    first, second = values.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


@torch.no_grad()
def last_prompt_attention(decoder_layer, hidden_states, position_embeddings,
                          attention_mask=None):
    """Return [batch, prompt_length] scores, averaged over query heads."""
    batch, length, _ = hidden_states.shape
    device = hidden_states.device
    positions = torch.arange(length, device=device).expand(batch, -1)
    valid = torch.ones((batch, length), dtype=torch.bool, device=device)
    if attention_mask is not None:
        if attention_mask.ndim != 2 or attention_mask.shape != (batch, length):
            raise ValueError("FastV scoring requires a 2D prefill padding mask")
        valid = attention_mask.to(device=device, dtype=torch.bool)
    last = positions.masked_fill(~valid, -1).max(dim=-1).values
    if (last < 0).any():
        raise ValueError("FastV cannot score an empty/padding-only prompt")

    attention = decoder_layer.self_attn
    normalized = decoder_layer.input_layernorm(hidden_states)
    rows = torch.arange(batch, device=device)
    query = attention.q_proj(normalized[rows, last].unsqueeze(1))
    key = attention.k_proj(normalized)
    query = query.view(batch, 1, attention.num_heads, attention.head_dim).transpose(1, 2)
    key = key.view(batch, length, attention.num_key_value_heads, attention.head_dim).transpose(1, 2)

    cos, sin = (item.to(device=device).expand(batch, -1, -1)
                for item in position_embeddings)
    query_cos = cos[rows, last][:, None, None, :]
    query_sin = sin[rows, last][:, None, None, :]
    query = query * query_cos + _rotate_half(query) * query_sin
    key = key * cos[:, None] + _rotate_half(key) * sin[:, None]
    key = key.repeat_interleave(attention.num_key_value_groups, dim=1)

    # This is a separate, read-only score calculation, not the decoder output.
    logits = torch.matmul(query.float(), key.float().transpose(-2, -1))
    logits = logits.squeeze(2) / math.sqrt(attention.head_dim)
    visible = valid & (positions <= last[:, None])
    config = attention.config
    if (getattr(config, "use_sliding_window", False)
            and getattr(config, "sliding_window", None) is not None
            and attention.layer_idx >= config.max_window_layers):
        visible &= positions > last[:, None] - config.sliding_window
    if not torch.isfinite(logits.masked_select(visible[:, None])).all():
        raise FloatingPointError(
            "Non-finite FastV Q/K scores in the FP16 model. Check the checkpoint "
            "and baseline for NaNs; precision is not silently changed."
        )
    logits = logits.masked_fill(~visible[:, None], float("-inf"))
    return torch.softmax(logits, dim=-1, dtype=torch.float32).mean(dim=1)
