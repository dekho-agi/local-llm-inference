"""KV cache arithmetic — the numbers that decide whether a context size fits.

Anchored to values measured on a real M5 Max, so a refactor that silently
changes the arithmetic fails here.
"""

import pytest

from llmctl.catalog import Model, kv_bytes_per_token

# Qwen3-Coder-Next: hybrid, only every 4th of 48 layers holds a KV cache.
QWEN3_NEXT = {
    "num_hidden_layers": 48,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "full_attention_interval": 4,
    "max_position_embeddings": 262144,
}

# Qwen3-Coder-30B: standard MoE, all 48 layers hold KV.
QWEN3_30B = {
    "num_hidden_layers": 48,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "max_position_embeddings": 262144,
}


def test_hybrid_counts_only_full_attention_layers():
    # 2 (K+V) * 2 heads * 256 dim * 2 bytes (bf16) * 12 layers
    assert kv_bytes_per_token(QWEN3_NEXT) == 24576


def test_dense_counts_every_layer():
    assert kv_bytes_per_token(QWEN3_30B) == 98304


def test_hybrid_is_cheaper_than_the_smaller_dense_model():
    """The counterintuitive result: the 80B costs 4x less per context token."""
    assert kv_bytes_per_token(QWEN3_30B) == 4 * kv_bytes_per_token(QWEN3_NEXT)


def test_head_dim_derived_when_absent():
    cfg = {
        "num_hidden_layers": 4,
        "num_key_value_heads": 2,
        "hidden_size": 1024,
        "num_attention_heads": 8,
    }
    # head_dim = 1024/8 = 128 -> 2*2*128*2*4
    assert kv_bytes_per_token(cfg) == 4096


@pytest.mark.parametrize("cfg", [{}, {"num_hidden_layers": 4}, {"num_key_value_heads": 2}])
def test_incomplete_config_returns_none(cfg):
    assert kv_bytes_per_token(cfg) is None


def test_model_totals_match_measured_values():
    m = Model(
        repo_id="mlx-community/Qwen3-Coder-Next-4bit",
        disk_gb=44.9,
        context=262144,
        kv_bytes_per_token=24576,
    )
    assert m.kv_gb() == 6.44  # measured 6.44 GB at full context
    assert m.total_gb() == 51.34  # and 51.3 GB with weights
    assert m.kv_gb(8192) == 0.2


def test_total_is_none_without_a_size():
    assert Model(repo_id="x", kv_bytes_per_token=1024, context=100).total_gb() is None


def test_agentic_requires_a_tool_parser():
    assert Model(repo_id="x", tool_parser="qwen3_coder").agentic
    assert not Model(repo_id="x", tool_parser=None).agentic


def test_servable_requires_mlx_lm_and_a_local_copy():
    assert Model(repo_id="x", runtime="mlx-lm", cached=True).servable
    assert not Model(repo_id="x", runtime="mlx-lm", cached=False).servable
    assert not Model(repo_id="x", runtime="mflux", cached=True).servable
