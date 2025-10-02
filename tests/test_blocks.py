# tests/test_blocks.py
import torch
import math
from src.models.blocks import TCN_block
from src.utils.constants import B


# ---- Hyperparams for the test (match your constants if you like) ----
T = 50    # time length 
BATCH = 2

def snr_db(x, y):
    num = (x**2).sum()
    den = ((x - y)**2).sum().clamp_min(1e-12)
    return 10.0 * torch.log10(num / den)

def test_single_block_shapes_and_length():
    x = torch.randn(BATCH, B, T)
    block = TCN_block()
    r, s = block(x)
    assert r.shape == (BATCH, B, T), f"Residual shape wrong: {r.shape}"
    assert s.shape == (BATCH, B, T), f"Skip shape wrong: {s.shape}"

def test_residual_is_input_plus_learned_update():
    torch.manual_seed(0)
    x = torch.randn(BATCH, B, T)
    block = TCN_block()

    with torch.no_grad():
        for m in block.modules():
            if isinstance(m, torch.nn.Conv1d):
                m.weight.zero_()
                if m.bias is not None:
                    m.bias.zero_()
        # single PReLU in this implementation
        if hasattr(block, 'prelu') and hasattr(block.prelu, 'weight'):
            block.prelu.weight.zero_()

    r, s = block(x)
    diff = (r - x).abs().max().item()
    assert diff < 1e-5, f"Residual path is not close to identity when weights are zeroed (max diff {diff})"
    assert s.abs().max().item() < 1e-5, "Skip output should be ~0 when weights are zeroed"

def test_causal_length_preservation():
    x = torch.randn(BATCH, B, T)
    block = TCN_block()
    r, s = block(x)
    assert r.size(2) == x.size(2), "Residual time length changed"
    assert s.size(2) == x.size(2), "Skip time length changed"

def test_mini_separator_accumulates_skips():
    X = 6
    x = torch.randn(BATCH, B, T)

    skip_sum = torch.zeros_like(x)
    r = x
    blocks = [TCN_block() for _ in range(X)]

    for i, blk in enumerate(blocks):
        r, s = blk(r)
        skip_sum = skip_sum + s
        assert r.shape == x.shape, f"Residual shape changed at block {i}"
        assert s.shape == x.shape, f"Skip shape changed at block {i}"

    assert skip_sum.shape == x.shape, "skip_sum has wrong shape after accumulation"

if __name__ == "__main__":
    # Simple CLI runner if you don't use pytest
    test_single_block_shapes_and_length()
    test_residual_is_input_plus_learned_update()
    test_causal_length_preservation()
    test_mini_separator_accumulates_skips()
    print("blocks.py tests passed ✅")
