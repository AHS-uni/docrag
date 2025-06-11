from typing import Any

from docrag.schema.model_settings import ImageProcessorConfig, TokenizerConfig


def apply_image_processor_config(
    image_processor: Any, config: ImageProcessorConfig
) -> None:
    image_processor.do_resize = config.resize
    image_processor.size = config.size
    if config.crop_strategy == "center":
        image_processor.do_center_crop = True
        image_processor.crop_size = config.size
        image_processor.do_random_crop = False
    elif config.crop_strategy == "random":
        image_processor.do_center_crop = False
        image_processor.do_random_crop = True
        image_processor.crop_size = config.size
    else:
        image_processor.do_center_crop = False
        image_processor.do_random_crop = False

    image_processor.do_rescale = config.rescale
    image_processor.rescale_factor = config.rescale_factor
    image_processor.do_normalize = config.normalize
    image_processor.image_mean = config.mean
    image_processor.image_std = config.std


def apply_tokenizer_config(tokenizer: Any, config: TokenizerConfig) -> None:
    tokenizer.padding_side = config.padding_side
    tokenizer.model_max_length = config.model_max_length
    if config.pad_token is not None:
        tokenizer.pad_token = config.pad_token
