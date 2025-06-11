from typing import Literal
from pydantic import BaseModel, Field

__all__ = ["GenerationConfig"]


class GenerationConfig(BaseModel):
    """
    Configuration for text generation tasks, mirroring Hugging Face's GenerationConfig.

    Attributes:
        max_length (int): The maximum total length of the generated sequence (prompt + new tokens).
        max_new_tokens (int | None): The maximum number of new tokens to generate, ignoring prompt length.
        min_length (int): The minimum total length of the generated sequence (prompt + new tokens).
        min_new_tokens (int | None): The minimum number of new tokens to generate, ignoring prompt length.
        early_stopping (bool | Literal['never']): Controls stopping condition for beam-based methods.
        max_time (float | None): Maximum allowed time for generation in seconds.
        stop_strings (str | list[str] | None): A string or list of strings that should terminate generation.
        do_sample (bool): Whether to use multinomial sampling (True) or greedy/beam search (False).
        num_beams (int): Number of beams for beam search; 1 disables beam search.
        num_beam_groups (int): Number of groups for diverse beam search.
        penalty_alpha (float): The balance penalty for contrastive search.
        top_k (int): Keep only the top_k highest probability tokens for sampling.
        top_p (float): Nucleus sampling cumulative probability threshold.
        temperature (float): Sampling temperature (>0).
        repetition_penalty (float): Penalty coefficient for repeated tokens (>0).
        length_penalty (float): Exponential penalty applied to sequence length in beam search.
        no_repeat_ngram_size (int): If >0, no n-gram of this size can be repeated.
    """

    max_length: int = Field(
        20, description="Maximum total sequence length (prompt + new tokens)."
    )
    max_new_tokens: int | None = Field(
        None,
        description="Maximum number of newly generated tokens, ignoring prompt length.",
    )
    min_length: int = Field(
        0, description="Minimum total sequence length (prompt + new tokens)."
    )
    min_new_tokens: int | None = Field(
        None,
        description="Minimum number of newly generated tokens, ignoring prompt length.",
    )
    early_stopping: bool | Literal["never"] = Field(
        False,
        description="Beam-stop rule: False=heuristic, True=stop when all beams finish, 'never'=strict algorithm.",
    )
    max_time: float | None = Field(
        None, description="Maximum generation time in seconds."
    )
    stop_strings: str | list[str] | None = Field(
        None, description="String or list of strings that should terminate generation."
    )

    do_sample: bool = Field(
        False, description="Whether to use sampling instead of greedy/beam search."
    )
    num_beams: int = Field(
        1, ge=1, description="Number of beams for beam search; 1 disables beam search."
    )
    num_beam_groups: int = Field(
        1, ge=1, description="Number of beam groups for diverse beam search."
    )
    penalty_alpha: float = Field(
        0.0, ge=0.0, description="Alpha penalty for contrastive search."
    )
    top_k: int = Field(
        50,
        ge=0,
        description="Number of highest-probability tokens to keep for top-k filtering.",
    )
    top_p: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling cumulative probability threshold.",
    )
    temperature: float = Field(1.0, gt=0.0, description="Sampling temperature (>0).")
    repetition_penalty: float = Field(
        1.0, gt=0.0, description="Penalty coefficient for repeated tokens (>0)."
    )
    length_penalty: float = Field(
        1.0,
        description="Exponential penalty applied to sequence length in beam search.",
    )
    no_repeat_ngram_size: int = Field(
        0, ge=0, description="If >0, no n-gram of this size can be repeated."
    )

    def to_kwargs(
        self, *, exclude_none: bool = True, exclude_defaults: bool = False
    ) -> dict[str, object]:
        """
        Prepare kwargs for `model.generate(...)`.

        Args:
            exclude_none (bool): If True, drop fields set to None.
            exclude_defaults (bool): If True, drop fields still equal to their default.

        Returns:
            dict[str, object]: Generation keyword arguments.
        """
        return self.model_dump(exclude_none=exclude_none, exclude_defaults=exclude_defaults)
