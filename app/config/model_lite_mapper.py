from enum import Enum


class ModelLiteMapper(Enum):
    """
    경량화 / 최적화 이미지 매핑
    """

    Bert_TRT= "aipaas-harbor.surromind.ai/trt-workflow/bert_trt_test:v0.5"

