import os
from PIL import Image
import webdataset as wds
from thinkdiff.datasets.datasets.base_dataset import BaseDataset
import torch
import json   # <-- 新增
from collections import deque
import random
import torch

MODALITY2ID = {"CT": 0, "X-ray": 1, "MRI": 2, "Ultrasound": 3, "Other": 4}
NUM_MODALITIES = len(MODALITY2ID)


def roundrobin_by_modality(data, num_modalities=5, key="json", history_size=200, maxlen=10):
    """
    完全优化版轮询平衡：
    - 所有FIFO缓冲区使用deque (O(1)复杂度)
    - 内存高效（decode前平衡）
    - 精确的剩余样本处理
    """
    # 使用deque替换列表，获得O(1)的popleft操作
    bufs = [deque(maxlen=maxlen) for _ in range(num_modalities)]  # 关键修正！
    history = [deque(maxlen=history_size) for _ in range(num_modalities)]
    modality_ptr = 0

    for sample in data:
        # 1. 从原始json获取modality信息
        json_data = sample.get(key) or sample.get("json") or sample[1]

        # 安全解析JSON
        if isinstance(json_data, bytes):
            try:
                json_data = json.loads(json_data.decode('utf-8', errors='ignore'))
            except:
                json_data = {}
        elif isinstance(json_data, str):
            try:
                json_data = json.loads(json_data)
            except:
                json_data = {}

        # 2. 获取modality_id
        modality = json_data.get("modality", "Other")
        mid = MODALITY2ID.get(modality, num_modalities - 1)
        if not (0 <= mid < num_modalities):
            mid = num_modalities - 1

        # 3. 添加到对应缓冲区 (O(1)操作)
        bufs[mid].append(sample)  # deque自动处理溢出，无需检查长度

        # 4. 轮询输出
        attempts = 0
        output = None

        while attempts < num_modalities and output is None:
            current_mid = (modality_ptr + attempts) % num_modalities

            if bufs[current_mid]:
                # O(1)复杂度的popleft
                output = bufs[current_mid].popleft()  # 关键修正！
                history[current_mid].append(output)

            elif history[current_mid]:
                # 从历史中随机选择
                output = random.choice(list(history[current_mid]))

            attempts += 1

        # 5. 更新指针
        if output is not None:
            modality_ptr = (current_mid + 1) % num_modalities
            yield output

    # 6. 清理剩余样本 - 这是必要的！
    for mid in range(num_modalities):
        # 按原始顺序输出（保持FIFO语义）
        while bufs[mid]:
            yield bufs[mid].popleft()  # O(1)操作


class CCSBUDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, location):
        super().__init__(vis_processor=vis_processor, text_processor=text_processor)
        # self._dbg = 0
        self.inner_dataset = wds.DataPipeline(
            wds.ResampledShards(location),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(1000, handler=wds.warn_and_continue),
            # TODO
            lambda data: roundrobin_by_modality(
                data,
                num_modalities=NUM_MODALITIES,
                key="json",
                history_size=300
            ),
            wds.decode("pilrgb", handler=wds.warn_and_continue),
            wds.to_tuple("jpg", "json", handler=wds.warn_and_continue),
            wds.map_tuple(self.vis_processor, handler=wds.warn_and_continue),
            wds.map(self.to_dict, handler=wds.warn_and_continue),

        )

    def to_dict(self, sample):

        img = sample[0]
        j = sample[1]  # python dict

        cap = j.get("caption", "") if isinstance(j, dict) else ""
        modality = j.get("modality", "Other") if isinstance(j, dict) else "Other"
        entities = j.get("entities", []) if isinstance(j, dict) else []

        mid = MODALITY2ID.get(modality, MODALITY2ID["Other"])
        entities_str = json.dumps(entities, ensure_ascii=False)

        out = {
            "image": img,
            "answer": self.text_processor(cap),  # 必须 raw_caption，保证实体 offset 不失效
            "entities": entities_str,  # List[str] in batch
            "modality": modality,  # List[str] in batch（可选）
            "modality_id": mid,  # 会被 collate 成 LongTensor[B]
        }

        return out
