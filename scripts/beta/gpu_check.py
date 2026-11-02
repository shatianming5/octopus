"""Trainer venv kernels on one GPU (scripts/beta: the GB200 check): flash-attn 2 varlen vs fp32 SDPA, and fla 0.4.2's gated
delta rule (chunk, fused recurrent) vs a plain torch fp32 recurrence (scale K^-0.5):
S <- S*exp(g_t); u = v_t - S^T k_t; S <- S + beta_t k_t u^T; o_t = scale * S^T q_t.
    CUDA_VISIBLE_DEVICES=3 ROOT/areal/.venv/bin/python gpu_check.py"""
import torch
import torch.nn.functional as F
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
print(torch.cuda.get_device_name(), torch.cuda.get_device_capability())
from flash_attn import flash_attn_varlen_func

torch.manual_seed(0)
q, k, v = (torch.randn(512, 8, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3))
cu = torch.tensor([0, 200, 512], dtype=torch.int32, device="cuda")
o = flash_attn_varlen_func(q, k, v, cu, cu, 312, 312, causal=True)
ref = torch.cat([F.scaled_dot_product_attention(*(x[a:b].transpose(0, 1)[None].float() for x in (q, k, v)),
                                                is_causal=True)[0].transpose(0, 1) for a, b in ((0, 200), (200, 512))])
print("flash_attn varlen bf16 max|diff| vs sdpa fp32", (o.float() - ref).abs().max().item())


def naive(q, k, v, g, beta):
    B, T, H, K = q.shape
    V = v.shape[-1]
    S = torch.zeros(B, H, K, V, device=q.device)
    out = torch.empty(B, T, H, V, device=q.device)
    for t in range(T):
        S = S * g[:, t].exp()[..., None, None]
        u = v[:, t] - torch.einsum("bhkv,bhk->bhv", S, k[:, t])
        S = S + beta[:, t][..., None, None] * torch.einsum("bhk,bhv->bhkv", k[:, t], u)
        out[:, t] = torch.einsum("bhkv,bhk->bhv", S, q[:, t]) * K ** -0.5
    return out


torch.manual_seed(0)
for T in (64, 300, 1024):
    B, H, K, V = 1, 4, 128, 128
    q = F.normalize(torch.randn(B, T, H, K, device="cuda"), dim=-1)
    k = F.normalize(torch.randn(B, T, H, K, device="cuda"), dim=-1)
    v = torch.randn(B, T, H, V, device="cuda")
    g = F.logsigmoid(torch.randn(B, T, H, device="cuda"))
    beta = torch.rand(B, T, H, device="cuda")
    ref = naive(q, k, v, g, beta)
    for name, fn in (("chunk", chunk_gated_delta_rule), ("recurrent", fused_recurrent_gated_delta_rule)):
        for dt in (torch.float32, torch.bfloat16):
            out, _ = fn(*(x.to(dt) for x in (q, k, v)), g=g, beta=beta.to(dt), output_final_state=False)
            d = (out.float() - ref).abs()
            print(f"T={T} {name} {str(dt)[6:]} max|diff| {d.max().item():.3g} mean {d.mean().item():.3g} (ref mean |o| {ref.abs().mean().item():.3g})")
