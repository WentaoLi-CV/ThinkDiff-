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
            wds.to_tuple("__key__", "jpg", "json", handler=wds.warn_and_continue),
            wds.map_tuple(
                lambda key: key,
                self.vis_processor,
                lambda meta: meta,
                handler=wds.warn_and_continue,
            ),
            wds.map(self.to_dict, handler=wds.warn_and_continue),
        )

    def to_dict(self, sample):
        sample_id = sample[2].get("sample_id", sample[0])
        return {
            "sample_id": sample_id,
            "image": sample[1],
            "answer": self.text_processor(sample[2]["caption"]),
            "entities": sample[2].get("entities", []),
            "modality": sample[2].get("modality"),
            "modality_id": sample[2].get("modality_id"),
        }
