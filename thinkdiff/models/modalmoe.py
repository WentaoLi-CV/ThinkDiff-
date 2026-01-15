# thinkdiff/models/modalmoe_linear.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from thinkdiff.models.moe_context import MoEContext

def transpose(weight, fan_in_fan_out: bool):
    return weight.T if fan_in_fan_out else weight


class LoraLayer:
    def __init__(self, r, lora_alpha, lora_dropout, merge_weights=True):
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
        self.merge_weights = merge_weights
        self.merged = False
        self.disable_adapters = False


class ModalMoELinear(nn.Linear, LoraLayer):
    def __init__(self, in_features, out_features, r=0, lora_alpha=1, lora_dropout=0.0,
                 fan_in_fan_out=False, merge_weights=True,
                 group_mode=False, num_modalities=0, shared_experts=0, modality_experts=0,
                 gate_embed_dim=0, gate_hidden_dim=128,
                 wspec_low=0.6, wspec_high=0.8, wspec_prior_weight=0.0,
                 lbc_weight=0.0, ctx=MoEContext,
                 enable_g5=False, g5_weight=0.0,
                 **kwargs):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        self.ctx = ctx

        self.fan_in_fan_out = fan_in_fan_out
        self.group_mode = bool(group_mode)
        self.num_modalities = int(num_modalities or 0)
        self.shared_experts = int(shared_experts or 0)
        self.modality_experts = int(modality_experts or 0)

        self.gate_embed_dim = int(gate_embed_dim or 0)
        self.gate_hidden_dim = int(gate_hidden_dim or 128)
        self.wspec_low = float(wspec_low)
        self.wspec_high = float(wspec_high)
        self.wspec_prior_weight = float(wspec_prior_weight or 0.0)

        self.lbc_weight = float(lbc_weight or 0.0)
        self.enable_g5 = bool(enable_g5)
        self.g5_weight = float(g5_weight or 0.0)

        if r > 0:
            if not self.group_mode:
                raise ValueError("Vision 模态建议直接 group_mode=True（ModalMoE）。")
            if self.num_modalities <= 0 or self.shared_experts <= 0 or self.modality_experts <= 0:
                raise ValueError("ModalMoE requires num_modalities/shared_experts/modality_experts > 0")

            self.lora_route_shared = nn.Linear(in_features, self.shared_experts, bias=False)
            self.lora_route_mod = nn.ModuleList([nn.Linear(in_features, self.modality_experts, bias=False)
                                                 for _ in range(self.num_modalities)])

            self.lora_A_shared = nn.ModuleList([nn.Linear(in_features, r, bias=False) for _ in range(self.shared_experts)])
            self.lora_B_shared = nn.ModuleList([nn.Linear(r, out_features, bias=False) for _ in range(self.shared_experts)])

            self.lora_A_mod = nn.ModuleList([
                nn.ModuleList([nn.Linear(in_features, r, bias=False) for _ in range(self.modality_experts)])
                for _ in range(self.num_modalities)
            ])
            self.lora_B_mod = nn.ModuleList([
                nn.ModuleList([nn.Linear(r, out_features, bias=False) for _ in range(self.modality_experts)])
                for _ in range(self.num_modalities)
            ])

            if self.gate_embed_dim > 0:
                self.lora_mod_emb = nn.Embedding(self.num_modalities, self.gate_embed_dim)
                gate_in = in_features + self.gate_embed_dim
            else:
                self.lora_mod_emb = None
                gate_in = in_features

            self.lora_group_gate = nn.Sequential(
                nn.Linear(gate_in, self.gate_hidden_dim, bias=True),
                nn.SiLU(),
                nn.Linear(self.gate_hidden_dim, 1, bias=True),
            )

            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False
            if self.bias is not None:
                self.bias.requires_grad = False

        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        self._reset_modalmoe_params_only()

    def _reset_modalmoe_params_only(self):
        if self.r <= 0:
            return
        nn.init.kaiming_uniform_(self.lora_route_shared.weight, a=math.sqrt(5))
        for m in range(self.num_modalities):
            nn.init.kaiming_uniform_(self.lora_route_mod[m].weight, a=math.sqrt(5))
        for i in range(self.shared_experts):
            nn.init.kaiming_uniform_(self.lora_A_shared[i].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_shared[i].weight)
        for m in range(self.num_modalities):
            for i in range(self.modality_experts):
                nn.init.kaiming_uniform_(self.lora_A_mod[m][i].weight, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B_mod[m][i].weight)

    def cv_squared(self, x):
        eps = 1e-10
        if x.numel() <= 1:
            return torch.zeros((), device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _lbc_loss(self, route_weight):
        importance = route_weight.sum(dim=(0, 1))
        top1 = route_weight.argmax(dim=-1)
        load = F.one_hot(top1, num_classes=route_weight.size(-1)).float().sum(dim=(0, 1))
        return self.cv_squared(importance) + self.cv_squared(load)

    def _cosine_sq(self, a, b):
        eps = 1e-8
        dot = (a * b).sum(dim=-1)
        na = a.norm(dim=-1).clamp_min(eps)
        nb = b.norm(dim=-1).clamp_min(eps)
        cos = dot / (na * nb)
        return (cos ** 2).mean()

    def forward(self, x):
        # x: (B,T,D) or (B,D)
        squeeze_2d = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze_2d = True

        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        if self.disable_adapters or self.r <= 0 or self.merged:
            return result.squeeze(1) if squeeze_2d else result

        modality_ids = self.ctx.modality_ids
        if modality_ids is None:
            raise ValueError("MoEContext.modality_ids is None (set before vision forward).")

        x_drop = self.lora_dropout(x)
        aux_loss = torch.zeros((), device=result.device, dtype=torch.float32)

        # ---- shared ----
        route_s = F.softmax(self.lora_route_shared(x), dim=-1, dtype=torch.float32).to(result.dtype)
        delta_s = torch.zeros_like(result)
        for i in range(self.shared_experts):
            delta_s = delta_s + route_s[..., i:i+1] * self.lora_B_shared[i](self.lora_A_shared[i](x_drop)) * self.scaling

        # ---- spec ----
        delta_m = torch.zeros_like(result)
        lbc_m_sum = torch.zeros((), device=result.device, dtype=torch.float32)
        tok_sum = torch.zeros((), device=result.device, dtype=torch.float32)

        # ✅ 关键：每个 m 都执行一次 routing + experts（保证参数参与图）
        for m in range(self.num_modalities):
            # (B,) bool -> (B,1,1) float mask
            mask_b = (modality_ids == m)
            mask = mask_b.to(result.dtype).view(-1, 1, 1)

            # 对全 batch 计算该模态路由（使 lora_route_mod[m] 参与图）
            route_m_all = F.softmax(self.lora_route_mod[m](x), dim=-1, dtype=torch.float32).to(result.dtype)

            # 对全 batch 计算该模态专家输出（使 lora_A_mod/B_mod[m][*] 参与图）
            delta_all = torch.zeros_like(result)
            for i in range(self.modality_experts):
                delta_all = delta_all + route_m_all[..., i:i + 1] * self.lora_B_mod[m][i](self.lora_A_mod[m][i](x_drop)) * self.scaling

            # 只让属于该模态的样本“生效”，其他样本乘 0
            delta_m = delta_m + delta_all * mask

            # LBC：只在该模态真实出现时统计（不影响 DDP）
            if self.lbc_weight != 0.0 and mask_b.any():
                idx = mask_b.nonzero(as_tuple=False).squeeze(-1)
                route_sel = route_m_all.index_select(0, idx).to(torch.float32)
                lbc_val = self._lbc_loss(route_sel)
                tok = torch.tensor(route_sel.size(0) * route_sel.size(1), device=result.device, dtype=torch.float32)
                lbc_m_sum = lbc_m_sum + lbc_val * tok
                tok_sum = tok_sum + tok

        # ---- gate ----
        pooled = x.mean(dim=1)
        if self.lora_mod_emb is not None:
            emb = self.lora_mod_emb(modality_ids)
            gate_in = torch.cat([pooled, emb], dim=-1)
        else:
            gate_in = pooled
        w_spec = torch.sigmoid(self.lora_group_gate(gate_in)).to(result.dtype)  # (B,1)
        w_spec_bt = w_spec.unsqueeze(1)

        delta = (1.0 - w_spec_bt) * delta_s + w_spec_bt * delta_m
        result = result + delta

        # ---- aux losses ----
        if self.lbc_weight != 0.0:
            lbc_s = self._lbc_loss(route_s.to(torch.float32))
            lbc_m = (lbc_m_sum / tok_sum.clamp_min(1.0)) if tok_sum.item() > 0 else torch.zeros_like(lbc_s)
            aux_loss = aux_loss + (lbc_s + lbc_m) * float(self.lbc_weight)

        if self.wspec_prior_weight != 0.0:
            w = w_spec.squeeze(-1).to(torch.float32)
            prior = F.relu(self.wspec_low - w).pow(2).mean() + F.relu(w - self.wspec_high).pow(2).mean()
            aux_loss = aux_loss + prior * float(self.wspec_prior_weight)

        if self.enable_g5 and (self.g5_weight != 0.0):
            comp = self._cosine_sq(delta_s.to(torch.float32), delta_m.to(torch.float32))
            aux_loss = aux_loss + comp * float(self.g5_weight)

        self.ctx.add_aux_loss(aux_loss)  # ✅ 累加到全局 context

        if squeeze_2d:
            result = result.squeeze(1)

        return result
