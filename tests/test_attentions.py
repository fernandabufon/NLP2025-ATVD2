# tests/test_attentions.py
import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from .adapters import run_attention_forward 
from basics.attentions import MultiHeadSelfAtt, GroupedQuerySelfAtt, MultiQuerySelfAtt

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

ATT_TYPES = ["mhsa", "gqa", "mqa"]

def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def _kwargs(att_type: str, n_head: int = 4, n_kv_head: int | None = None):
    kw = {"n_head": n_head}
    if att_type == "gqa":
        kw["n_kv_head"] = n_kv_head if n_kv_head is not None else 2
    return kw


# ------------------------------------------------------------
# Contratos básicos
# ------------------------------------------------------------
@pytest.mark.parametrize("att_type", ATT_TYPES)
def test_preserves_shape_dtype_device(att_type):
    """
    O que testa:
      - A função forward preserva (B,T,C), dtype e device.

    Por que importa:
      - Quebra de shape ou dtype é uma fonte comum de bugs em atenção multi-head,
        especialmente em reshape/transpose/merge de heads e projeções.
    """
    device = _device()
    torch.manual_seed(0)
    x = torch.randn(3, 7, 32, device=device, dtype=torch.float32)

    y = run_attention_forward(att_type, x, **_kwargs(att_type))
    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert y.device == x.device


def test_rejects_non_3d_input():
    """
    O que testa:
      - O adapter recusa entradas que não sejam 3D (B,T,C), com erro claro.

    Por que importa:
      - Evita uso indevido (p.ex., esquecer dimensão temporal) e garante mensagem de erro
        amigável para avaliação automática em massa.
    """
    x = torch.randn(10, 16)  # 2D de propósito
    with pytest.raises(ValueError):
        run_attention_forward("mhsa", x, n_head=4)


@pytest.mark.parametrize("att_type", ATT_TYPES)
def test_backward_and_no_nans(att_type):
    """
    O que testa:
      - O backward() roda sem erros e a saída não contém NaN/Inf.

    Por que importa:
      - Confirma a conectividade computacional com autograd e estabilidade numérica mínima;
        NaN/Inf podem surgir com máscaras aditivas em meia-precisão ou problemas de escala.
    """
    torch.manual_seed(0)
    x = torch.randn(2, 6, 32, requires_grad=True)
    y = run_attention_forward(att_type, x, **_kwargs(att_type))
    loss = y.pow(2).mean()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("att_type", ATT_TYPES)
def test_requires_emb_divisible_by_heads(att_type):
    """
    O que testa:
      - Falha quando C (n_embd) não é divisível por n_head.

    Por que importa:
      - A divisão por cabeças requer partição exata do canal; evitar isso captura
        bugs de configuração (p.ex., H que não divide C).
    """
    x = torch.randn(2, 5, 30)  # 30 % 8 != 0
    with pytest.raises(AssertionError):
        _ = run_attention_forward(att_type, x, **_kwargs(att_type, n_head=8))


# ------------------------------------------------------------
# Determinismo / Dtypes
# ------------------------------------------------------------
@pytest.mark.parametrize("att_type", ATT_TYPES)
def test_deterministic_with_seed(att_type):
    """
    O que testa:
      - Mesmo seed + mesmo input ⇒ mesma saída.

    Por que importa:
      - Reprodutibilidade é crucial para corrigir e comparar implementações dos alunos.

    Nota técnica:
      - É importante consumir a MESMA sequência de RNG antes da init dos pesos. Por isso,
        após resetar a seed, reamostramos x2 em vez de clonar x1.
    """
    torch.manual_seed(123)
    x1 = torch.randn(2, 5, 32)
    y1 = run_attention_forward(att_type, x1, **_kwargs(att_type))

    torch.manual_seed(123)
    x2 = torch.randn(2, 5, 32)  # reamostra, consumindo o mesmo RNG antes da init dos pesos
    y2 = run_attention_forward(att_type, x2, **_kwargs(att_type))

    assert torch.allclose(y1, y2, atol=0, rtol=0)


# ------------------------------------------------------------
# Correção matemática (toy) — MHSA vs fórmula manual
# ------------------------------------------------------------
def _mhsa_manual(q, k, v):
    """
    Implementação manual do SDPA (para verificação numérica).
    q,k,v: (B, H, T, hs)
    
    O que faz:
      - Computa att = softmax((QKᵀ)/√d_k) e out = att @ V.

    Por que importa:
      - Compara saída da implementação com uma referência "manual" em escala pequena,
        detectando bugs sutis de reshape/transpose/scale/softmax.
    """
    B, H, T, hs = q.shape
    scale = 1.0 / math.sqrt(hs)
    att = torch.matmul(q, k.transpose(-2, -1)) * scale              # (B,H,T,T)
    att = torch.softmax(att, dim=-1)
    out = torch.matmul(att, v)                                      # (B,H,T,hs)
    return out

def test_mhsa_matches_manual_sdpa_toy():
    """
    O que testa:
      - Em um caso pequeno, MHSA deve coincidir com a fórmula manual do SDPA,
        usando os MESMOS pesos internos da camada.

    Por que importa:
      - Valida end-to-end (projeções Q/K/V, split por head, SDPA, merge e proj final),
        eliminando diferenças numéricas via reuso dos pesos.
    """
    torch.manual_seed(0)
    B, T, C, H = 2, 4, 16, 4
    hs = C // H
    x = torch.randn(B, T, C)

    # Instancia MHSA e força máscara=None/causal=False
    class Cfg: pass
    cfg = Cfg(); cfg.n_embd=C; cfg.n_head=H
    layer = MultiHeadSelfAtt(cfg)

    # Executa uma passada para obter Q,K,V do próprio layer (reusando os pesos)
    with torch.no_grad():
        qkv = layer.c_attn(x)
    q, k, v = qkv.split(C, dim=2)  # (B,T,C) cada
    q = q.reshape(B, T, H, hs).transpose(1, 2)
    k = k.reshape(B, T, H, hs).transpose(1, 2)
    v = v.reshape(B, T, H, hs).transpose(1, 2)

    out_manual = _mhsa_manual(q, k, v)                              # (B,H,T,hs)
    y_ref = out_manual.transpose(1, 2).contiguous().reshape(B, T, C)
    y_ref = layer.c_proj(y_ref)                                     # proj final

    y = layer(x)  # implementação testada
    assert torch.allclose(y, y_ref, atol=1e-5, rtol=1e-5)


# ------------------------------------------------------------
# Formas e contagem EXATA de parâmetros
# ------------------------------------------------------------
def _n_params(module: nn.Module) -> int:
    # Soma todas as entradas paramétricas (pesos + bias)
    return sum(p.numel() for p in module.parameters())

def test_param_shapes_and_counts_exact():
    """
    O que testa:
      - Shapes de pesos e contagem EXATA de parâmetros para MHSA, GQA e MQA.

    Por que importa:
      - Garante que GQA/MQA são de fato mais econômicas que MHSA, além de pegar bugs
        de dimensionamento incorreto em projeções Q/K/V e proj final.
    """
    class Cfg: pass
    C, H, Hkv = 256, 8, 2
    hs = C // H

    # MHSA
    cfg = Cfg(); cfg.n_embd=C; cfg.n_head=H
    mhsa = MultiHeadSelfAtt(cfg)
    # Shapes esperadas
    assert mhsa.c_attn.weight.shape == (3*C, C)
    assert mhsa.c_proj.weight.shape == (C, C)
    # Contagem exata (pesos+bias): (c_attn) + (c_proj)
    n_mhsa = (3*C*C + 3*C) + (C*C + C)

    # GQA
    cfg = Cfg(); cfg.n_embd=C; cfg.n_head=H; cfg.n_kv_head=Hkv
    gqa = GroupedQuerySelfAtt(cfg)
    Ckv = Hkv * hs
    assert gqa.q_proj.weight.shape == (C, C)
    assert gqa.k_proj.weight.shape == (Ckv, C)
    assert gqa.v_proj.weight.shape == (Ckv, C)
    assert gqa.c_proj.weight.shape == (C, C)
    n_gqa = (C*C + C) + (Ckv*C + Ckv) + (Ckv*C + Ckv) + (C*C + C)

    # MQA
    cfg = Cfg(); cfg.n_embd=C; cfg.n_head=H
    mqa = MultiQuerySelfAtt(cfg)
    assert mqa.q_proj.weight.shape == (C, C)
    assert mqa.k_proj.weight.shape == (hs, C)
    assert mqa.v_proj.weight.shape == (hs, C)
    assert mqa.c_proj.weight.shape == (C, C)
    n_mqa = (C*C + C) + (hs*C + hs) + (hs*C + hs) + (C*C + C)

    assert _n_params(mhsa) == n_mhsa
    assert _n_params(gqa) == n_gqa
    assert _n_params(mqa) == n_mqa

    # Economia esperada
    assert n_gqa < n_mhsa
    assert n_mqa < n_mhsa
