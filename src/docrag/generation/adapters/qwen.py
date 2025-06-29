import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

from ..adapter import Adapter
from ..registry import register

__all__ = [
    "QwenAdapter",
]


@register("qwen2.5-vl")
@register("qwen")
class QwenAdapter(Adapter):
    """
    Adapter for the Qwen2.5-VL vision-language  model.
    """

    def _load(self) -> None:
        model_config = self.config.model
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_config.path,
            torch_dtype=getattr(torch, model_config.dtype),
            device_map=model_config.device_map,
            trust_remote_code=model_config.trust_remote_code,
            cache_dir=model_config.cache_dir,
            low_cpu_mem_usage=model_config.low_cpu_mem_usage,
            attn_implementation=model_config.attn_implementation,
        )
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            model_config.path,
            trust_remote_code=model_config.trust_remote_code,
            cache_dir=model_config.cache_dir,
            use_fast=True,
        )

        image_processor_config = self.config.image_processor
        image_processor_kwargs = image_processor_config.to_kwargs(exclude_defaults=True)
        for k, v in image_processor_kwargs.items():
            setattr(self.processor.image_processor, k, v)

        tokenizer_config = self.config.tokenizer
        tokenizer_kwargs = tokenizer_config.to_kwargs(exclude_defaults=True)
        for k, v in tokenizer_kwargs.items():
            setattr(self.processor.tokenizer, k, v)

    def generate(
        self,
        images: list[Image.Image],
        text: str,
    ) -> str:
        prompt = self._apply_prompt_template(text)

        system_prompt = self.config.system_prompt
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": image} for image in images],
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        chat = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[chat],
            images=images,
            return_tensors="pt",
        )
        inputs.to(self.model.device, dtype=self.model.dtype)

        generation_config = self.config.generation
        generation_kwargs = generation_config.to_kwargs(exclude_defaults=True)
        outputs = self.model.generate(**inputs, **generation_kwargs)

        input_ids = inputs["input_ids"]
        generated_ids = outputs[0][input_ids.shape[-1] :]

        return self.processor.decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
