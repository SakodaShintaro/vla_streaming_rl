from typing import Dict, List, Optional

import torch
from torch import Tensor, nn

from vla_streaming_rl.simlingo.utils.custom_types import DrivingExample, DrivingInput


class WaypointInputAdaptor(nn.Module):
    """
    Takes an input of shape [B, N, 2] and returns an output of shape [B, N, token_size]
    Args:
        token_size: feature dimension of output tensor.
        hidden_size: hidden dimension used in Linear layers under the hood.
    """

    def __init__(
        self,
        token_size: int = 258,
        hidden_size: int = 64,
        hidden_size2: int = 128,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size2),
            nn.ReLU(True),
            nn.Linear(hidden_size2, token_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input with dims [B, N, 2]

        Returns:
            Output with dims [B, N, token_size]
        """
        x = self.mlp(x)
        return x


class DrivingAdaptor(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        mlp_dim=256,
    ):
        super().__init__()
        self.heads = {}
        self.order = []

        self.future_waypoints = 20
        self.query_embeds_wps = nn.Parameter(
            0.02 * torch.randn((1, self.future_waypoints, hidden_size))
        )
        self.route_head = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim * 2),
            nn.SiLU(True),
            nn.Linear(mlp_dim * 2, mlp_dim),
            nn.SiLU(True),
            nn.Linear(mlp_dim, 2, bias=False),
        )

        self.queries = {"route": self.query_embeds_wps}
        self.sizes = {"route": self.future_waypoints}
        self.heads["route"] = self.route_head
        self.order.append("route")

        dim = 2
        self.future_speed_waypoints = 10  # TODO: read from config
        self.query_embeds_speed = nn.Parameter(
            0.02 * torch.randn((1, self.future_speed_waypoints, hidden_size))
        )
        self.speed_wps_head = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim), nn.SiLU(True), nn.Linear(mlp_dim, dim, bias=False)
        )
        self.heads["speed_wps"] = self.speed_wps_head
        self.queries["speed_wps"] = self.query_embeds_speed
        self.sizes["speed_wps"] = self.future_speed_waypoints
        self.order.append("speed_wps")

    def forward(self, driving_input: DrivingInput, **kwargs) -> Dict[str, Tensor]:
        b = driving_input.camera_images.shape[0]
        inputs = torch.cat(
            [self.queries[input_type].expand(b, -1, -1) for input_type in self.order], dim=1
        )
        inputs_mask = torch.ones_like(inputs[:, :, 0], dtype=torch.bool)

        return {"inputs": inputs, "inputs_mask": inputs_mask}

    def get_predictions(self, features: Tensor, logits: Optional[Tensor] = None) -> Dict:

        current_index = 0
        predictions = {}
        for input_type in self.order:
            size = self.sizes[input_type]

            head = self.heads[input_type]
            feature = features[:, current_index : current_index + size]
            # The waypoint heads may be promoted to float32 (RL fine-tuning)
            # while the VLM features stay bfloat16; cast to the head's dtype
            # so both the frozen-backbone forward and float32 training work.
            feature = feature.to(head[0].weight.dtype)
            prediction = head(feature).cumsum(1)

            predictions[input_type] = prediction
            current_index += size

        return predictions


class LanguageAdaptor(nn.Module):
    def __init__(self, language_model):
        super().__init__()
        self.embed_tokens = language_model.model.embed_tokens
        if hasattr(language_model.model, "lm_head"):
            self.lm_head = language_model.model.lm_head
        elif hasattr(language_model.model, "embed_out"):
            self.lm_head = language_model.model.embed_out
        elif hasattr(language_model.model.base_model.model, "output"):
            self.lm_head = language_model.model.base_model.model.output
        else:
            raise ValueError("Language model must have `lm_head` or `embed_out` attribute.")

    def forward(self, driving_input: DrivingInput, inference=False, **kwargs) -> Dict[str, Tensor]:
        label = driving_input.prompt_inference if inference else driving_input.prompt

        ids = label.phrase_ids.long()
        ids_valid = label.phrase_valid  # true => is fed into model
        ids_mask = label.loss_masking  # true => takes part in loss

        inputs = self.embed_tokens(ids.clamp(min=0, max=self.embed_tokens.num_embeddings - 1))
        return {"inputs": inputs, "inputs_mask": ids_valid, "_ids": ids, "_ids_mask": ids_mask}


class AdaptorList(nn.Module):
    """
    Each adaptor is responsible for converting a driving example
    to a sequence of tokens and computing the loss on the token outputs.
    Adaptors are only used during training.
    """

    def __init__(
        self,
        driving: Optional[DrivingAdaptor] = None,
        language: Optional[LanguageAdaptor] = None,
    ):
        super().__init__()
        self.driving = driving
        self.language = language

    @property
    def adaptors(self):
        dct: Dict[str, DrivingAdaptor | LanguageAdaptor] = {}
        if self.language is not None:
            dct["language"] = self.language
        if self.driving is not None:
            dct["driving"] = self.driving
        return dct

    def forward(self, example: DrivingExample, **kwargs) -> Dict[str, Tensor]:
        """
        Construct input embeddings for the given driving example.
        """

        input_dict: Dict[str, Tensor] = {}
        inputs_list: List[Tensor] = []
        inputs_mask_list: List[Tensor] = []

        for key, adaptor in self.adaptors.items():
            adaptor_input_dict = adaptor.forward(example, **kwargs)
            inputs_list.append(adaptor_input_dict["inputs"])
            inputs_mask_list.append(adaptor_input_dict["inputs_mask"])
            input_dict.update({key + "_" + k: v for k, v in adaptor_input_dict.items()})

        inputs = torch.cat(inputs_list, dim=1)
        inputs_mask = torch.cat(inputs_mask_list, dim=1)
        split_sizes = torch.as_tensor([x.size(1) for x in inputs_list])
        arange = torch.arange(inputs.size(0), device=inputs.device)[:, None]

        # Apply random permutation of modalities during training
        rand_perm = torch.arange(inputs.size(1), device=inputs.device).expand(inputs.size(0), -1)
        # Apply permutation to move invalid tokens to end of sequence
        valid_perm = (
            inputs_mask[arange, rand_perm].byte().argsort(dim=-1, descending=True, stable=True)
        )
        perm = rand_perm.gather(1, valid_perm)

        input_dict["inputs"] = inputs[arange, perm]
        input_dict["inputs_mask"] = inputs_mask[arange, perm]
        input_dict["perm"] = perm
        input_dict["split_sizes"] = split_sizes
        return input_dict
