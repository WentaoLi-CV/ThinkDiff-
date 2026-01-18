import os
import logging
import warnings

from thinkdiff.common.registry import registry
from thinkdiff.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from thinkdiff.datasets.datasets.laion_dataset import LaionDataset
from thinkdiff.datasets.datasets.cc_sbu_dataset import CCSBUDataset
from thinkdiff.datasets.datasets.cc_sbu_dataset_mllama_vllm_process_wids import CCSBUMllamaVllmProcessDatasetWids
from thinkdiff.datasets.datasets.llava_instruct_dataset_mllama_embed_2 import LlavaInstructMllamaEmbedDataset_2
from thinkdiff.datasets.datasets.medical_webdataset import MedicalWebDataset

@registry.register_builder("cc_sbu")
class CCSBUBuilder(BaseDatasetBuilder):
    train_dataset_cls = CCSBUDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/cc_sbu/defaults.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            location=build_info.storage,
        ).inner_dataset

        return datasets
    

@registry.register_builder("medical_webdataset")
class MedicalWebDatasetBuilder(BaseDatasetBuilder):
    train_dataset_cls = MedicalWebDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/medical_webdataset/defaults.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            location=build_info.storage,
        ).inner_dataset

        return datasets


@registry.register_builder("laion")
class LaionBuilder(BaseDatasetBuilder):
    train_dataset_cls = LaionDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/laion/defaults.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            location=build_info.storage,
        ).inner_dataset

        return datasets


@registry.register_builder("cc_sbu_mllama_vllm_process_wids")
class CCSBUMllamaVllmEmbedDatasetBuilder(BaseDatasetBuilder):
    train_dataset_cls = CCSBUMllamaVllmProcessDatasetWids

    DATASET_CONFIG_DICT = {"default": "configs/datasets/cc_sbu_mllama_vllm_process_wids/defaults.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        dataset_ins = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            location=build_info.storage,
        )
        datasets[split] = dataset_ins.inner_dataset
        datasets[split].collater = dataset_ins.collater

        return datasets
        


@registry.register_builder("llava_instruct_mllama_embed_2")
class LlavaInstructMllamaEmbedDataset_2_Builder(BaseDatasetBuilder):
    train_dataset_cls = LlavaInstructMllamaEmbedDataset_2

    DATASET_CONFIG_DICT = {"default": "configs/datasets/llava_instruct_mllama_embed_2/defaults.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        dataset_ins = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            location=build_info.storage,
            build_info=build_info,
        )
        datasets[split] = dataset_ins.inner_dataset
        datasets[split].collater = dataset_ins.collater

        return datasets
    
