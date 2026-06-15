"""Build SimLingo's per-frame ``LanguageLabel`` prompt for ``DrivingInput``.

Extracted from ``LingoAgent._tick``. Phases:

1. Phrase the target as a waypoint with a ``<TARGET_POINT>`` placeholder
   filled by the ego-frame coords.
2. Wrap with the current speed and the CoT / non-CoT suffix.
3. Apply the InternLM2 chat template and expand the ``<image>`` token to
   the model's image-token sequence.
4. Tokenize into a ``LanguageLabel``.

Single-sample (batch=1) only — agent_simlingo runs one frame at a time.
``config.eval_route_as`` is asserted to be ``"target_point"`` (the only
mode currently exercised); the legacy ``"command"`` and
``"target_point_command"`` branches were dead at runtime.
"""

import numpy as np

from vla_streaming_rl.simlingo.simlingo_training.models.encoder.internvl2_vendored import (
    conversation as conv_module,
)
from vla_streaming_rl.simlingo.utils.custom_types import LanguageLabel


class PromptBuilder:
    _IMG_START_TOKEN = "<img>"
    _IMG_END_TOKEN = "</img>"
    _IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
    _PATCHES_PER_IMAGE = 2

    def __init__(self, config, tokenizer, num_image_token, device):
        assert config.eval_route_as == "target_point", (
            f"PromptBuilder only supports eval_route_as='target_point'; "
            f"got {config.eval_route_as!r}"
        )
        self._config = config
        self._tokenizer = tokenizer
        self._num_image_token = num_image_token
        self._device = device

    def build(self, speed, ego_target_point, ego_next_target_point):
        target_phrase, placeholder_batch_list = self._target_phrase(
            ego_target_point, ego_next_target_point
        )
        prompt = self._wrap_speed_and_cot(speed, target_phrase)
        prompt_str = self._apply_chat_template(prompt)
        return self._tokenize(prompt_str, placeholder_batch_list)

    def _target_phrase(self, ego_target_point, ego_next_target_point):
        target_points_np = np.array([ego_target_point, ego_next_target_point])
        token_id = self._tokenizer.convert_tokens_to_ids("<TARGET_POINT>")
        placeholder_batch_list = [{token_id: target_points_np}]
        return "Target waypoint: <TARGET_POINT><TARGET_POINT>.", placeholder_batch_list

    def _wrap_speed_and_cot(self, speed, target_phrase):
        if self._config.use_cot:
            return f"Current speed: {speed} m/s. {target_phrase} What should the ego do next?"
        return f"Current speed: {speed} m/s. {target_phrase} Predict the waypoints."

    def _apply_chat_template(self, prompt):
        question = "<image>\n" + prompt
        template = conv_module.get_conv_template("internlm2-chat")
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)

        query = template.get_prompt()
        system_prompt = (
            template.system_template.replace("{system_message}", template.system_message)
            + template.sep
        )
        query = query.replace(system_prompt, "")

        image_tokens = (
            self._IMG_START_TOKEN
            + self._IMG_CONTEXT_TOKEN * self._num_image_token * self._PATCHES_PER_IMAGE
            + self._IMG_END_TOKEN
        )
        return query.replace("<image>", image_tokens, 1)

    def _tokenize(self, prompt_str, placeholder_batch_list):
        prompt_batch_list = [prompt_str]
        tokenized = self._tokenizer(
            prompt_batch_list,
            padding=True,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        ids = tokenized["input_ids"]
        valid = ids != self._tokenizer.pad_token_id
        return LanguageLabel(
            phrase_ids=ids.to(self._device),
            phrase_valid=valid.to(self._device),
            phrase_mask=valid.to(self._device),
            placeholder_values=placeholder_batch_list,
            language_string=prompt_batch_list,
            loss_masking=None,
        )
