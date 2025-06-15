"""
DocRAG demo – concrete generator backends & factory.
"""

import torch
from typing import List, Optional
from PIL import Image

from .registry import GENERATORS
from .base import Generator


@GENERATORS.register("internvl")
class InternVLGenerator(Generator):
    """Generator wrapper for OpenGVLab/InternVL3-1B-hf.

    Args:
        device: Optional device specifier, e.g. "cuda:0" or "cpu".
    """

    name = "internvl"
    model_name = "OpenGVLab/InternVL3-2B-hf"

    def __init__(self, device: Optional[str] = None):
        """Load the processor and model onto the specified device."""
        from transformers import (
            AutoProcessor as InternProcessor,
            AutoModelForImageTextToText as InternModel,
        )

        self.processor = InternProcessor.from_pretrained(self.model_name, use_fast=True)
        self.model = InternModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=device or "auto",
        ).eval()

    @torch.inference_mode()
    def generate(
        self,
        query: str,
        images: Optional[List[Image.Image]] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate an answer from text and optional images, with an optional system prompt.

        Args:
            query: The user’s question.
            images: List of PIL.Image page images, or None for text-only.
            system_prompt: Optional system-level instruction.

        Returns:
            The model’s generated answer as a string.
        """
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": image} for image in (images or [])],
                    {"type": "text", "text": query},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)

        prompt_length = inputs["input_ids"].shape[-1]

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
        )
        trimmed = outputs[0, prompt_length:]
        decoded = self.processor.decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return decoded


@GENERATORS.register("qwenvl")
class QwenVLGenerator(Generator):
    """Generator wrapper for OpenGVLab/Qwen2-VL-Chat-1.5B.

    Args:
        device: Optional device specifier, e.g. "cuda:0" or "cpu".
    """

    name = "qwen"
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"

    def __init__(self, device: Optional[str] = None):
        """Load the processor and model onto the specified device."""
        from transformers import (
            AutoProcessor as QwenProcessor,
            Qwen2_5_VLForConditionalGeneration as QwenModel,
        )

        self.processor = QwenProcessor.from_pretrained(self.model_name, use_fast=True)
        self.model = QwenModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=device or "auto",
        ).eval()

    @torch.inference_mode()
    def generate(
        self,
        query: str,
        images: list[Image.Image] | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """
        Generate an answer from text (and optional images), with an optional system prompt.

        Args:
            query: The user’s question.
            images: List of PIL.Image page images, or None for text-only.
            system_prompt: Optional system-level instruction.

        Returns:
            The model’s generated answer as a string.
        """
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": image} for image in (images or [])],
                    {"type": "text", "text": query},
                ],
            },
        ]

        chat = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=chat,
            images=images,
            return_tensors="pt",
        ).to(self.model.device)

        prompt_length = inputs["input_ids"].shape[-1]

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
        )
        trimmed = outputs[0, prompt_length:]
        decoded = self.processor.decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return decoded


def get_generator(name: str, device: Optional[str] = None) -> Generator:
    """
    Load (if needed) and return a generator instance.

    Args:
        name: Key of the generator backend (e.g. "internvl-3b").
        device: Optional device specifier for model placement.

    Returns:
        An instance of the requested Generator.
    """
    return GENERATORS.load(name, device)
