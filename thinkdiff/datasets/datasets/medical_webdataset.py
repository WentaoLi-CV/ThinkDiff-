import json
import webdataset as wds
from thinkdiff.datasets.datasets.base_dataset import BaseDataset
import random
from collections import deque
import torch
from torch.utils.data._utils.collate import default_collate

MODALITY2ID = {"CT": 0, "X-ray": 1, "MRI": 2, "Ultrasound": 3}
NUM_MODALITIES = len(MODALITY2ID)

def balanced_batch_by_modality(
    data,
    batch_size: int,
    num_modalities: int = NUM_MODALITIES,
    per_modality: int = 1,
    key: str = "modality_id",
    max_queue: int = 256,
    history_size: int = 300,
):
    assert batch_size == num_modalities * per_modality, \
        f"Need batch_size == num_modalities*per_modality, got {batch_size}"

    queues = [deque(maxlen=max_queue) for _ in range(num_modalities)]
    history = [deque(maxlen=history_size) for _ in range(num_modalities)]

    def _get_mid(sample):
        try:
            mid = sample.get(key, None)
            if isinstance(mid, torch.Tensor):
                mid = int(mid.item())
            else:
                mid = int(mid)
        except Exception:
            mid = num_modalities - 1
        if not (0 <= mid < num_modalities):
            mid = num_modalities - 1
        return mid

    for sample in data:
        mid = _get_mid(sample)
        queues[mid].append(sample)
        history[mid].append(sample)

        while True:
            # 能否凑齐每个模态 per_modality 个
            ready = True
            for m in range(num_modalities):
                if len(queues[m]) < per_modality and len(history[m]) == 0:
                    ready = False
                    break
            if not ready:
                break

            batch = []
            for m in range(num_modalities):
                for _ in range(per_modality):
                    if queues[m]:
                        batch.append(queues[m].popleft())
                    else:
                        batch.append(random.choice(list(history[m])))
            yield default_collate(batch)


class MedicalWebDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, location, batch_size: int = 4, balance_per_batch: bool = False,
                 per_modality: int = 1, num_modalities: int = NUM_MODALITIES):
        super().__init__(vis_processor=vis_processor, text_processor=text_processor)

        pipe = wds.DataPipeline(
            wds.ResampledShards(location),
            wds.split_by_node,
            wds.split_by_worker,
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(1000, handler=wds.warn_and_continue),
            wds.decode("pilrgb", handler=wds.warn_and_continue),
            wds.to_tuple("jpg", "json", handler=wds.warn_and_continue),
            wds.map_tuple(self.vis_processor, handler=wds.warn_and_continue,),
            wds.map(self.to_dict, handler=wds.warn_and_continue),
        )

        if balance_per_batch:
            pipe = pipe.compose(lambda data: balanced_batch_by_modality(
                data,
                batch_size=batch_size,
                num_modalities=num_modalities,
                per_modality=per_modality,
                key="modality_id",
            ))
            pipe.already_batched = True

        self.inner_dataset = pipe

    def to_dict(self, sample):
        return {
            "image": sample[0],
            "answer": self.text_processor(sample[1]["caption"]),
            "entities": json.dumps(sample[1].get("entities", []), ensure_ascii=False),
            "modality": sample[1].get("modality"),
            "modality_id": sample[1].get("modality_id"),
        }
