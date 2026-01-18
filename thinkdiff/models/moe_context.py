import torch
from typing import Optional


class MoEContext:
    def __init__(self, num_modalities: int):
        self.num_modalities = int(num_modalities)
        self.modality_ids: Optional[torch.LongTensor] = None
        self._aux_loss = None  # torch.float32 scalar on device

    def set_modality_ids(self, modality_ids: Optional[torch.Tensor], device: torch.device):
        if modality_ids is None:
            self.modality_ids = None
        else:
            mid = modality_ids.to(device=device, dtype=torch.long)
            mid = mid.clamp_(0, self.num_modalities - 1)
            self.modality_ids = mid
        # reset aux each forward
        self._aux_loss = torch.zeros((), device=device, dtype=torch.float32)

    def add_aux_loss(self, loss: torch.Tensor):
        if loss is None:
            return
        self._aux_loss = self._aux_loss + loss.to(dtype=torch.float32)

    def pop_aux_loss(self) -> torch.Tensor:
        if self._aux_loss is None:
            return torch.zeros((), dtype=torch.float32)
        out = self._aux_loss
        self._aux_loss = torch.zeros_like(out)
        return out
