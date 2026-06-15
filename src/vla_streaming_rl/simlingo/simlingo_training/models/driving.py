from typing import Tuple

import pytorch_lightning as pl
import torch
from torch import Tensor

from vla_streaming_rl.simlingo.simlingo_training.models.adaptors import (
    AdaptorList,
    DrivingAdaptor,
    LanguageAdaptor,
    WaypointInputAdaptor,
)
from vla_streaming_rl.simlingo.simlingo_training.models.llm import LLM
from vla_streaming_rl.simlingo.simlingo_training.models.vlm import VLMEncoderModel
from vla_streaming_rl.simlingo.simlingo_training.utils.custom_types import (
    DrivingInput,
    DrivingOutput,
)


class DrivingModel(pl.LightningModule):
    def __init__(
        self,
        cfg_data_module,
        processor,
        cache_dir,
        **cfg,
    ):
        super().__init__()
        self.save_hyperparameters()

        for key, value in cfg.items():
            setattr(self, key, value)

        self.processor = processor

        self.cfg_data_module = cfg_data_module

        # ``self.vision_model`` / ``self.language_model`` are DictConfig
        # nodes loaded from the checkpoint's ``.hydra/config.yaml``.
        # Upstream used ``hydra.utils.instantiate(_target_=...)`` to pick
        # the class — we always want VLMEncoderModel / LLM here, so
        # construct them directly. Drops ``_target_`` from the kwargs
        # since these classes take ``**cfg`` and don't expect that key.
        vm_kwargs = {k: v for k, v in self.vision_model.items() if k != "_target_"}
        self.vision_model = VLMEncoderModel(
            cfg_data_module=cfg_data_module,
            processor=self.processor,
            **vm_kwargs,
        )

        lm_kwargs = {k: v for k, v in self.language_model.items() if k != "_target_"}
        self.language_model = LLM(cache_dir=cache_dir, **lm_kwargs)

        self.adaptors = AdaptorList(
            language=LanguageAdaptor(self.language_model),
            driving=DrivingAdaptor(self.language_model.hidden_size),
        )

        self.wp_encoder = WaypointInputAdaptor(
            token_size=self.language_model.hidden_size,
            hidden_size=256,
            hidden_size2=512,
        )

        if "tokenizer" in self.processor.__dict__:
            self.tokenizer = self.processor.tokenizer
        else:
            self.tokenizer = self.processor

    # ------------------------------------------------------------------
    # Policy head
    # ------------------------------------------------------------------
    @property
    def policy_head(self) -> DrivingAdaptor:
        """Maps the encoder's driving-query features to waypoint/route predictions."""
        return self.adaptors.driving

    # ------------------------------------------------------------------
    # Encoder + policy head (inference)
    # ------------------------------------------------------------------
    def forward(self, driving_input: DrivingInput) -> DrivingOutput:
        """
        Samples a trajectory from the model.

        Two stages: the **encoder** (VLM image features + LLM, with
        autoregressive language generation) produces per-batch driving
        features, then the **policy head** maps those features to the
        waypoint/route predictions.

        Returns ``(speed_wps, route, language, driving_features)``. The
        last element is the (B, 30, hidden) slice of the LLM's last
        hidden state at the waypoint-query positions — i.e. the input
        to the policy head's waypoint MLPs. It's exposed so downstream RL
        wrappers can use it as a state vector for a critic without
        having to re-run the VLM.
        """
        # Encoder: embed the prompt and splice in the VLM image features.
        prompt_embeds, prompt_masks = self._encode_prompt(driving_input)

        # Decode each batch item separately (padding differs per item) and
        # collect the per-item outputs.
        language = []
        route_per_item, speed_wps_per_item, features_per_item = [], [], []
        for b_idx, (prompt_embed, prompt_mask) in enumerate(zip(prompt_embeds, prompt_masks)):
            sampled_tokens, driving_features, driving_logits = self._generate_driving_features(
                prompt_embed.unsqueeze(0), prompt_mask.unsqueeze(0), driving_input, b_idx
            )

            # Policy head: features -> {"route": ..., "speed_wps": ...}.
            predictions = self.policy_head.get_predictions(driving_features, driving_logits)
            route_per_item.append(predictions["route"])
            speed_wps_per_item.append(predictions["speed_wps"])
            features_per_item.append(driving_features)

            language.append(
                self.tokenizer.batch_decode(sampled_tokens, skip_special_tokens=True)[0]
            )

        # Concatenate across batch items -> leading dim B.
        route = torch.cat(route_per_item, dim=0)
        speed_wps = torch.cat(speed_wps_per_item, dim=0)
        driving_features = torch.cat(features_per_item, dim=0)

        return speed_wps, route, language, driving_features

    def _encode_prompt(self, driving_input: DrivingInput) -> Tuple[Tensor, Tensor]:
        """
        Embed the inference prompt and splice the VLM image features into the
        placeholder tokens. Returns ``(input_embeds, attention_mask)`` for the
        full batch — the language-model input before generation.
        """
        adaptor_dict = self.vision_model.image_encoder.replace_placeholder_tokens(
            adaptor_dict=self.adaptors(driving_input, inference=True),
            pixel_values=driving_input.camera_images,
            placeholder_values=driving_input.prompt_inference.placeholder_values,
            wp_encoder=self.wp_encoder,
        )
        return adaptor_dict["language_inputs"], adaptor_dict["language_inputs_mask"]

    def _generate_driving_features(
        self,
        prompt_embed: Tensor,
        prompt_mask: Tensor,
        driving_input: DrivingInput,
        b_idx: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        For a single batch item: autoregressively generate the language answer,
        then run one more forward pass with the driving-query tokens appended.
        Returns ``(sampled_tokens, driving_features, driving_logits)`` where the
        features/logits cover only the driving-query positions.
        """
        # BUG: input_embeds, cot
        sampled_tokens, prompt_embed = self.language_model.greedy_sample(
            prompt_embed,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=100,
            input_embed_matrix=self.adaptors.language.embed_tokens.weight,
            logit_matrix=self.adaptors.language.lm_head.weight,
            attention_mask=prompt_mask,
            # position_ids=position_ids,
        )

        driving_inputs = self.policy_head(driving_input)
        input_embed_concat = torch.cat(
            (prompt_embed, driving_inputs["inputs"][b_idx].unsqueeze(0)), dim=1
        )
        features, logits = self.language_model.forward(input_embed_concat)

        len_driving = driving_inputs["inputs"].size(1)
        return sampled_tokens, features[:, -len_driving:], logits[:, -len_driving:]
