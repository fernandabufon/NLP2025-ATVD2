# adapters.py
import torch
from basics.attentions import MultiHeadSelfAtt, GroupedQuerySelfAtt, MultiQuerySelfAtt

def run_attention_forward(att_type: str, x: torch.Tensor, n_head: int, n_kv_head: int | None = None):
    """
    Executa um forward de atenção, escolhendo a variante pelo att_type.
    att_type: "mhsa" | "gqa" | "mqa"
    x: tensor (B, T, C) C define n_embd automaticamente
    n_head: número de cabeças de Q (sempre obrigatório)
    n_kv_head: obrigatório apenas para GQA (Grouped-Query)
    """
    if x.dim() != 3:
        raise ValueError(f"x deve ser 3D (B,T,C); recebido: {tuple(x.shape)}")

    # Monta um config mínimo do jeito que as classes esperam
    class _Cfg: 
        pass

    cfg = _Cfg()
    cfg.n_embd = x.size(-1)
    cfg.n_head = n_head

    if att_type == "gqa":
        if n_kv_head is None:
            raise ValueError("n_kv_head é obrigatório para GQA")
        cfg.n_kv_head = n_kv_head

    if att_type == "mhsa":
        layer = MultiHeadSelfAtt(cfg)
    elif att_type == "gqa":
        layer = GroupedQuerySelfAtt(cfg)
    elif att_type == "mqa":
        layer = MultiQuerySelfAtt(cfg)
    else:
        raise ValueError(f"Tipo de atenção desconhecido: {att_type}")

    layer = layer.to(device=x.device, dtype=x.dtype)
    
    return layer(x)
