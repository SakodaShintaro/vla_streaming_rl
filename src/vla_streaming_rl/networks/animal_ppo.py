# SPDX-License-Identifier: MIT
"""The Animal-AI Olympics winning network, ported from ``~/work/rl_animal``.

A visual trunk -- the original Fixup residual tower with channel attention, or
any of the pretrained encoders in ``networks/modules/image_processor.py``, which
is where both now live -- a small dense branch for the velocity/clock vector, and
a recurrent cell shared by the policy and value heads. It is deliberately not a
:class:`NetworkInterface`: that contract is built around a replay batch, while
this network is driven by an on-policy PPO rollout, so it stays a plain
``nn.Module`` and :class:`AnimalPPOAgent` owns the loss.

The one change from the original is the image format: it takes the CHW float
observation this repo's wrappers already produce, instead of converting uint8
NHWC itself.
"""

import torch
import torch.nn.functional as F
from torch import nn

from vla_streaming_rl.networks.modules.image_processor import ImageProcessor

ACTION_NUM = 9
HIDDEN_NODES = 1024
VELS_HIDDEN = 128
TEMPORAL_UNITS = 512
LAYER_NORM_EPSILON = 1e-5
TEMPORAL_MODEL_TYPES = ("lstm", "gru")


class LayerNormLSTMCell(nn.Module):
    """The gates are ordered i, f, o, u, the input and recurrent contributions
    are normalized separately over the whole ``4 * units`` axis, and the cell
    state is normalized again before the output gate."""

    def __init__(self, input_size: int, units: int) -> None:
        super().__init__()
        self.units = units
        self.state_size = 2 * units
        self.wx = nn.Parameter(torch.zeros(input_size, 4 * units))
        self.gx = nn.Parameter(torch.ones(4 * units))
        self.bx = nn.Parameter(torch.zeros(4 * units))
        self.wh = nn.Parameter(torch.zeros(units, 4 * units))
        self.gh = nn.Parameter(torch.ones(4 * units))
        self.bh = nn.Parameter(torch.zeros(4 * units))
        self.b = nn.Parameter(torch.zeros(4 * units))
        self.gc = nn.Parameter(torch.ones(units))
        self.bc = nn.Parameter(torch.zeros(units))

    @staticmethod
    def normalize(x: torch.Tensor, gain: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        variance = x.var(dim=1, unbiased=False, keepdim=True)
        return (x - mean) / torch.sqrt(variance + LAYER_NORM_EPSILON) * gain + bias

    def forward(self, inputs: list, state: torch.Tensor, masks: list) -> tuple:
        """``inputs`` and ``masks`` are sequences of ``(sequence_num, ...)`` tensors;
        ``state`` is the ``(sequence_num, 2 * units)`` pair carried between batches. A
        mask of 1 means the previous step ended an episode, so the state is
        zeroed before that step is consumed."""
        cell, hidden = torch.split(state, self.units, dim=1)
        outputs = []
        for x, mask in zip(inputs, masks):
            keep = (1.0 - mask).unsqueeze(1)
            cell = cell * keep
            hidden = hidden * keep
            z = (
                self.normalize(x.matmul(self.wx), self.gx, self.bx)
                + self.normalize(hidden.matmul(self.wh), self.gh, self.bh)
                + self.b
            )
            i, f, o, u = torch.split(z, self.units, dim=1)
            cell = torch.sigmoid(f) * cell + torch.sigmoid(i) * torch.tanh(u)
            hidden = torch.sigmoid(o) * torch.tanh(self.normalize(cell, self.gc, self.bc))
            outputs.append(hidden)

        return outputs, torch.cat([cell, hidden], dim=1)


class SequenceGRUCell(nn.Module):
    """``torch.nn.GRUCell`` behind the same interface as
    :class:`LayerNormLSTMCell`: a sequence in, the per-step hiddens and the
    carried state out. There is no LayerNorm variant here -- the point is to use
    PyTorch's cell as it is -- so only the episode masking and the loop are ours.
    """

    def __init__(self, input_size: int, units: int) -> None:
        super().__init__()
        self.units = units
        self.state_size = units
        self.cell = nn.GRUCell(input_size, units)

    def forward(self, inputs: list, state: torch.Tensor, masks: list) -> tuple:
        """``inputs`` and ``masks`` are sequences of ``(sequence_num, ...)`` tensors;
        ``state`` is the ``(sequence_num, units)`` hidden carried between batches. A
        mask of 1 means the previous step ended an episode, so the state is
        zeroed before that step is consumed."""
        hidden = state
        outputs = []
        for x, mask in zip(inputs, masks):
            hidden = self.cell(x, hidden * (1.0 - mask).unsqueeze(1))
            outputs.append(hidden)

        return outputs, hidden


def build_temporal_model(temporal_model_type: str, input_size: int, units: int) -> nn.Module:
    assert temporal_model_type in TEMPORAL_MODEL_TYPES, (
        f"Unknown temporal_model_type: {temporal_model_type!r} "
        f"(expected one of {TEMPORAL_MODEL_TYPES})"
    )
    if temporal_model_type == "lstm":
        return LayerNormLSTMCell(input_size, units)
    return SequenceGRUCell(input_size, units)


class AnimalBackbone(nn.Module):
    """Everything the winning network is below its heads: the visual trunk, the
    dense branch and the recurrent cell. It is a class of its own so a network
    with other heads -- see ``networks/animal_actor_critic.py`` -- reuses this
    body verbatim instead of copying it.

    Two things are configurable. The trunk is an
    :class:`ImageProcessor`, so ``image_encoder_type`` picks the original Fixup
    tower (``"fixup"``, trained from scratch) or one of the frozen pretrained
    encoders beside it in ``networks/modules/image_processor.py``. Whatever it
    produces is flattened into the single visual token ``visual_hidden`` reads,
    so ``image_encode_mode = "single_token"`` pools the image the way that
    encoder was pretrained to instead of handing the dense layer a patch grid.
    And ``temporal_model_type`` picks the recurrence: the original LayerNorm
    LSTM (``"lstm"``) or PyTorch's ``nn.GRUCell`` (``"gru"``), whose state is a
    single hidden rather than a (cell, hidden) pair -- hence ``init_state``
    asking the cell for its width instead of writing ``2 * TEMPORAL_UNITS``.
    """

    def __init__(
        self,
        observation_space_shape: tuple[int, ...],
        vels_size: int,
        image_encoder_type: str,
        image_encoder_output_dim: int,
        image_encode_mode: str,
        image_encoder_trainable: bool,
        temporal_model_type: str,
    ) -> None:
        """``observation_space_shape`` is the (C, H, W) of the image observation and
        ``vels_size`` the width of the velocity/clock vector; both come from what
        the environment produces, so neither is written out here."""
        super().__init__()
        self.image_processor = ImageProcessor(
            observation_space_shape,
            image_encoder_type,
            image_encoder_output_dim,
            image_encode_mode,
            image_encoder_trainable,
        )
        channels, height, width = self.image_processor.output_shape
        self.flat_size = channels * height * width

        self.vels_hidden = nn.Linear(vels_size, VELS_HIDDEN)
        self.visual_hidden = nn.Linear(self.flat_size, HIDDEN_NODES)
        self.joint_hidden = nn.Linear(VELS_HIDDEN + HIDDEN_NODES, HIDDEN_NODES)
        self.temporal_model = build_temporal_model(
            temporal_model_type, HIDDEN_NODES, TEMPORAL_UNITS
        )

    def init_state(self, sequence_num: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(sequence_num, self.temporal_model.state_size, device=device)

    def features(self, visual: torch.Tensor) -> torch.Tensor:
        """``visual`` is (B, C, H, W) float, as the wrappers produce it."""
        out = self.image_processor.encode(visual)
        # channels last before flattening: the order the dense layer's weights expect
        return out.permute(0, 2, 3, 1).reshape(out.shape[0], -1)

    def embed(self, visual: torch.Tensor, vels: torch.Tensor) -> tuple:
        """The per-step visual latent the dense branch produces and the joint
        hidden the recurrent cell reads. Split out of ``forward`` so a head that needs the
        visual latent -- see ``networks/animal_world_critic.py`` -- reads it off
        the same pass instead of running the trunk twice."""
        visual_latent = F.elu(self.visual_hidden(self.features(visual)))
        hidden = torch.cat([F.elu(self.vels_hidden(vels)), visual_latent], dim=-1)
        return visual_latent, F.elu(self.joint_hidden(hidden))

    def recurrent(
        self, hidden: torch.Tensor, state: torch.Tensor, dones: torch.Tensor, sequence_num: int
    ) -> tuple:
        steps_num = hidden.shape[0] // sequence_num
        sequence = hidden.reshape(sequence_num, steps_num, -1).unbind(dim=1)
        mask_sequence = dones.to(hidden.dtype).reshape(sequence_num, steps_num).unbind(dim=1)
        outputs, temporal_state = self.temporal_model(sequence, state, mask_sequence)
        return torch.stack(outputs, dim=1).reshape(-1, TEMPORAL_UNITS), temporal_state

    def forward(
        self,
        visual: torch.Tensor,
        vels: torch.Tensor,
        state: torch.Tensor,
        dones: torch.Tensor,
        sequence_num: int,
    ) -> tuple:
        """The batch is sequence-major: the entry for sequence s at step t sits
        at ``s * steps_num + t``, which is what the recurrent unrolling and the
        sequence slicing both assume. Acting is one sequence of one step, the
        update is one per minibatch window. Returns the ``(sequence_num * steps_num,
        TEMPORAL_UNITS)`` recurrent output and the state carried out of the batch."""
        _, hidden = self.embed(visual, vels)
        return self.recurrent(hidden, state, dones, sequence_num)


class AnimalPPONetwork(AnimalBackbone):
    """The backbone plus the two PPO heads: categorical logits and a state value."""

    def __init__(
        self,
        observation_space_shape: tuple[int, ...],
        vels_size: int,
        image_encoder_type: str,
        image_encoder_output_dim: int,
        image_encode_mode: str,
        image_encoder_trainable: bool,
        temporal_model_type: str,
    ) -> None:
        super().__init__(
            observation_space_shape,
            vels_size,
            image_encoder_type,
            image_encoder_output_dim,
            image_encode_mode,
            image_encoder_trainable,
            temporal_model_type,
        )
        self.value_head = nn.Linear(TEMPORAL_UNITS, 1)
        self.logits_head = nn.Linear(TEMPORAL_UNITS, ACTION_NUM)

    def forward(
        self,
        visual: torch.Tensor,
        vels: torch.Tensor,
        state: torch.Tensor,
        dones: torch.Tensor,
        sequence_num: int,
    ) -> tuple:
        temporal_out, temporal_state = super().forward(visual, vels, state, dones, sequence_num)
        return self.logits_head(temporal_out), self.value_head(temporal_out), temporal_state

    def forward_for_update(
        self,
        visual: torch.Tensor,
        vels: torch.Tensor,
        state: torch.Tensor,
        dones: torch.Tensor,
        actions: torch.Tensor,
        sequence_num: int,
    ) -> tuple:
        """The update pass, in the shape the PPO loss reads: the policy logits,
        the state values, and whatever auxiliary loss the network carries beyond
        PPO's own -- nothing here, the world-critic terms in
        ``networks/animal_world_critic.py``, and the scalar each reports.

        ``actions`` is what separates it from ``forward``: an auxiliary objective
        may be action-conditioned, while the heads themselves never are.
        """
        del actions
        logits, value, _ = self(visual, vels, state, dones, sequence_num)
        return logits, value.squeeze(-1), torch.zeros((), device=value.device), {}
