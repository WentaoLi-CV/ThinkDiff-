# thinkdiff/models/inject_modalmoe.py
from __future__ import annotations
from typing import Iterable, List, Tuple, Dict, Any
import torch.nn as nn
from .modalmoe_linear import ModalMoELinear
from .moe_context import MoEContext


def _get_parent_and_child(root: nn.Module, full_name: str):
    if "." in full_name:
        parent_name, child_name = full_name.rsplit(".", 1)
        parent = root.get_submodule(parent_name)
    else:
        parent = root
        child_name = full_name
    return parent, child_name


def _is_bnb_linear(m: nn.Module) -> bool:
    # bitsandbytes quantized linear often named Linear4bit / Linear8bitLt
    name = m.__class__.__name__.lower()
    return ("linear4bit" in name) or ("linear8bit" in name) or ("8bitlt" in name)


# TODO
# def inject_modalmoe_linear(
#     root: nn.Module,
#     ctx: MoEContext,
#     target_modules: Iterable[str] = ("fc1", "fc2"),
#     module_name_filter: str | None = None,  # optional regex/substring filter
#     **modalmoe_kwargs,
#     ) -> List[str]:
#     """
#     Replace matched nn.Linear with ModalMoELinear in-place.
#     Matching rule:
#       - name endswith any in target_modules (or ".<key>")
#       - optional module_name_filter as substring
#     """
#     replaced: List[str] = []
#     all_named = list(root.named_modules())
#
#     for name, m in all_named:
#         if not isinstance(m, nn.Linear):
#             continue
#         if isinstance(m, ModalMoELinear):
#             continue
#         if module_name_filter is not None and (module_name_filter not in name):
#             continue
#         if not any(name.endswith(k) or name.endswith("." + k) for k in target_modules):
#             continue
#
#         parent, child = _get_parent_and_child(root, name)
#         bias = (m.bias is not None)
#
#         new_m = ModalMoELinear(
#             m.in_features,
#             m.out_features,
#             bias=bias,
#             ctx=ctx,
#             **modalmoe_kwargs,
#         )
#
#         # keep pretrained weights (share param object like LoRAMoE does)
#         new_m.weight = m.weight
#         if bias:
#             new_m.bias = m.bias
#
#         # move to correct device (dtype handled later globally)
#         new_m.to(device=m.weight.device)
#
#         setattr(parent, child, new_m)
#         replaced.append(name)
#
#     return replaced
def inject_modalmoe_linear(
    root: nn.Module,
    ctx: MoEContext,
    target_modules: Iterable[str] = ("fc1", "fc2"),
    module_name_filter: str | None = None,
    allow_quantized_linear: bool = False,   # 默认 False：量化视觉会破坏“共享权重对象”的实现
    **modalmoe_kwargs,
) -> List[str]:
    """
    Replace matched nn.Linear with ModalMoELinear in-place.
    NOTE:
      - This injector shares base weight param object (new_m.weight = m.weight),
        so it requires m to be a standard nn.Linear.
      - If vision backbone is quantized (bnb Linear4bit/8bitLt), injection should be disabled
        unless you also refactor ModalMoELinear to wrap base_layer instead of sharing weight.
    """
    replaced: List[str] = []
    all_named = list(root.named_modules())

    for name, m in all_named:
        if isinstance(m, ModalMoELinear):
            continue

        # only consider Linear-like
        if not isinstance(m, nn.Linear):
            if _is_bnb_linear(m):
                if not allow_quantized_linear:
                    raise RuntimeError(
                        f"[ModalMoE inject] Found quantized linear layer at '{name}' ({m.__class__.__name__}). "
                        "Your current ModalMoE injector shares nn.Linear.weight; this is incompatible with bnb quantized "
                        "linears. Solution: do NOT quantize vision backbone (recommended), i.e. set quant_scope='language'. "
                        "If you insist on quantizing vision, you must refactor ModalMoELinear to wrap base_layer."
                    )
                else:
                    # allow_quantized_linear=True 也仍然不安全，除非你同步改了 ModalMoELinear
                    # 这里直接 continue 避免 silent wrong behavior
                    continue
            continue

        if module_name_filter is not None and (module_name_filter not in name):
            continue
        if not any(name.endswith(k) or name.endswith("." + k) for k in target_modules):
            continue

        parent, child = _get_parent_and_child(root, name)
        bias = (m.bias is not None)

        new_m = ModalMoELinear(
            m.in_features,
            m.out_features,
            bias=bias,
            ctx=ctx,
            **modalmoe_kwargs,
        )

        # keep pretrained weights (share param object like LoRAMoE)
        new_m.weight = m.weight
        if bias:
            new_m.bias = m.bias

        new_m.to(device=m.weight.device)
        setattr(parent, child, new_m)
        replaced.append(name)

    return replaced
