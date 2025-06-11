import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from ..adapter import Adapter
from ..registry import register
from ..utils import apply_tokenizer_config, apply_image_processor_config


@register("internvl3")
@register("internvl")
class InternVLAdapter(Adapter):
    """
    Adapter for the InternVL3 vision-language  model.
    """

    def _load(self) -> None:
        model_config = self.settings.model
        image_processor_config = self.settings.image_processor
        tokenizer_config = self.settings.tokenizer

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_config.path,
            torch_dtype=getattr(torch, model_config.dtype),
            device_map=model_config.device_map,
            trust_remote_code=model_config.trust_remote_code,
            cache_dir=model_config.cache_dir,
            low_cpu_mem_usage=model_config.low_cpu_mem_usage,
        )
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            model_config.path,
            trust_remote_code=model_config.trust_remote_code,
            cache_dir=model_config.cache_dir,
            use_fast=True,
        )

        apply_image_processor_config(
            self.processor.image_processor, image_processor_config
        )
        apply_tokenizer_config(self.processor.tokenizer, tokenizer_config)

    def generate(
        self,
        images: list[Image.Image],
        text: str,
    ) -> str:
        prompt = self._apply_prompt_template(text)

        system_prompt = self.settings.system_prompt
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

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device, dtype=self.model.dtype)

        generation_config = self.settings.generate
        outputs = self.model.generate(**inputs, **generation_config.to_diff_dict())

        input_ids = inputs["input_ids"]
        generated_ids = outputs[0][input_ids.shape[-1] :]

        return self.processor.decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
