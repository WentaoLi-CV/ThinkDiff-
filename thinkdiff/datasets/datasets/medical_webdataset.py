import json

import webdataset as wds

from thinkdiff.datasets.datasets.base_dataset import BaseDataset


class MedicalWebDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, location):
        super().__init__(vis_processor=vis_processor, text_processor=text_processor)

        self.inner_dataset = wds.DataPipeline(
            wds.ResampledShards(location),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(1000, handler=wds.warn_and_continue),
            wds.decode("pilrgb", handler=wds.warn_and_continue),
            wds.to_tuple("jpg", "json", handler=wds.warn_and_continue),
            wds.map_tuple(self.vis_processor, handler=wds.warn_and_continue,),
            wds.map(self.to_dict, handler=wds.warn_and_continue),
        )

    def to_dict(self, sample):
        return {
            "sample_id": sample[0],
            "image": sample[1],
            "answer": self.text_processor(sample[2]["caption"]),
            "entities": json.dumps(sample[2].get("entities", []), ensure_ascii=False),
            "modality": sample[2].get("modality"),
            "modality_id": sample[2].get("modality_id"),
        }
