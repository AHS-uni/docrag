from PIL import Image
import torch
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

from docrag.schema.inputs import GeneratorInput
from docrag.utils import Timer

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

    def load(self) -> None:
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

    def generate(self, input: GeneratorInput) -> tuple[str, float, int]:
        with Timer() as timer:
            prompt = self._apply_prompt_template(input.text)
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.config.system_prompt}],
                },
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": image} for image in input.images],
                        {"type": "text", "text": prompt},
                    ],
                },
            ]

            chat = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            model_inputs = self.processor(
                text=[chat],
                images=input.images,
                return_tensors="pt",
            ).to(self.model.device)

            prompt_length = model_inputs["input_ids"].shape[1]

            generation_config = self.config.generation
            generation_kwargs = generation_config.to_kwargs(exclude_defaults=True)

            with torch.inference_mode():
                outputs = self.model.generate(**model_inputs, **generation_kwargs)

                generated_ids = outputs[0, prompt_length:]
                token_count = generated_ids.shape[-1]
                decoded_text = self.processor.decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )

        return decoded_text, timer.elapsed, token_count

    def batch_generate(self, batch_input: list[GeneratorInput]) -> list[tuple[str, float, int]]:
        with Timer() as timer:
            batch_chat: list[str] = []
            batch_images: list[list[Image.Image]] = []
            for input in batch_input:
                prompt = self._apply_prompt_template(input.text)
                messages = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": self.config.system_prompt}],
                    },
                    {
                        "role": "user",
                        "content": [
                            *[{"type": "image", "image": image} for image in input.images],
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                batch_chat.append(
                    self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True,
                    )
                )
                batch_images.append(input.images)

            batch_model_inputs = self.processor(
                text=batch_chat,
                images=batch_images,
                return_tensors="pt",
                padding=True,
            ).to(self.model.device)

            prompt_length = batch_model_inputs["input_ids"].shape[-1]

            generation_config = self.config.generation
            generation_kwargs = generation_config.to_kwargs(exclude_defaults=True)

            batch_count_tokens: list[int] = []
            batch_decoded_text: list[str] = []

            with torch.inference_mode():
                outputs = self.model.generate(**batch_model_inputs, **generation_kwargs)

                for i in range(len(batch_input)):
                    generated_ids = outputs[i, prompt_length:]
                    token_count = generated_ids.shape[-1]
                    decoded_text = self.processor.decode(
                        generated_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                    )
                    batch_count_tokens.append(token_count)
                    batch_decoded_text.append(decoded_text)


        return [
            (decoded_text, timer.elapsed, token_count)
            for decoded_text, token_count in zip(batch_decoded_text, batch_count_tokens)
        ]
