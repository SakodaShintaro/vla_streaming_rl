# SPDX-License-Identifier: MIT
"""The Animal-AI Olympics winning network, ported from ``~/work/rl_animal``.

A Fixup residual tower with channel attention, a small dense branch for the
velocity/clock vector, and a LayerNorm LSTM shared by the policy and value
heads. It is deliberately not a :class:`NetworkInterface`: that contract is
built around a replay batch, while this network is driven by an on-policy PPO
rollout, so it stays a plain ``nn.Module`` and :class:`AnimalPPOAgent` owns the
loss.

The one change from the original is the image format: it takes the CHW float
observation this repo's wrappers already produce, instead of converting uint8
NHWC itself.
"""

import torch
import torch.nn.functional as F
from torch import nn

ACTION_NUM = 9
DEPTHS = (16, 32, 64, 128)
HIDDEN_NODES = 1024
VELS_HIDDEN = 128
LSTM_UNITS = 512
LAYER_NORM_EPSILON = 1e-5


class ChannelAttention(nn.Module):
    """The channel means through two bias-free 1x1 convolutions, squashed to a
    per-channel gate."""

    def __init__(self, depth: int) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(depth, depth // 4, 1, bias=False)
        self.expand = nn.Conv2d(depth // 4, depth, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.mean(dim=(2, 3), keepdim=True)
        out = self.expand(F.elu(self.reduce(out)))
        return torch.sigmoid(out)


class FixupAttentionBlock(nn.Module):
    """A residual block with no normalization, four scalar biases and a scalar
    multiplier, and the channel gate applied between the two convolutions."""

    def __init__(self, depth: int) -> None:
        super().__init__()
        self.res1 = nn.Conv2d(depth, depth, 3, padding=1, bias=False)
        self.res2 = nn.Conv2d(depth, depth, 3, padding=1, bias=False)
        self.attention = ChannelAttention(depth)
        self.bias0 = nn.Parameter(torch.zeros(()))
        self.bias1 = nn.Parameter(torch.zeros(()))
        self.bias2 = nn.Parameter(torch.zeros(()))
        self.bias3 = nn.Parameter(torch.zeros(()))
        self.multiplier = nn.Parameter(torch.ones(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.elu(x) + self.bias0
        out = self.res1(out) + self.bias1
        out = out * self.attention(out)
        out = F.elu(out) + self.bias2
        out = self.res2(out) * self.multiplier + self.bias3
        return out + x


class LayerNormLSTMCell(nn.Module):
    """The gates are ordered i, f, o, u, the input and recurrent contributions
    are normalized separately over the whole ``4 * units`` axis, and the cell
    state is normalized again before the output gate."""

    def __init__(self, input_size: int, units: int) -> None:
        super().__init__()
        self.units = units
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
        """``inputs`` and ``masks`` are sequences of ``(env_num, ...)`` tensors;
        ``state`` is the ``(env_num, 2 * units)`` pair carried between batches. A
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


class AnimalBackbone(nn.Module):
    """Everything the winning network is below its heads: the residual tower, the
    dense branch and the LSTM. It is a class of its own so a network with other
    heads -- see ``networks/animal_actor_critic.py`` -- reuses this body verbatim
    instead of copying it; ``AnimalPPONetwork`` extends it rather than holding
    one so the parameter names, and therefore existing checkpoints, are unchanged.
    """

    def __init__(self, observation_space_shape: tuple[int, ...], vels_size: int) -> None:
        """``observation_space_shape`` is the (C, H, W) of the image observation and
        ``vels_size`` the width of the velocity/clock vector; both come from what
        the environment produces, so neither is written out here."""
        super().__init__()
        in_channels, height, width = observation_space_shape

        self.tower = nn.ModuleList()
        for depth in DEPTHS:
            self.tower.append(
                nn.ModuleDict(
                    {
                        "conv": nn.Conv2d(in_channels, depth, 3, padding=1),
                        "block1": FixupAttentionBlock(depth),
                        "block2": FixupAttentionBlock(depth),
                    }
                )
            )
            in_channels = depth

        # one stride-2 max pool per stage, rounding up
        for _ in DEPTHS:
            height = -(-height // 2)
            width = -(-width // 2)
        self.flat_size = height * width * DEPTHS[-1]

        self.vels_hidden = nn.Linear(vels_size, VELS_HIDDEN)
        self.visual_hidden = nn.Linear(self.flat_size, HIDDEN_NODES)
        self.joint_hidden = nn.Linear(VELS_HIDDEN + HIDDEN_NODES, HIDDEN_NODES)
        self.lstm = LayerNormLSTMCell(HIDDEN_NODES, LSTM_UNITS)

    def init_state(self, env_num: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(env_num, 2 * LSTM_UNITS, device=device)

    def features(self, visual: torch.Tensor) -> torch.Tensor:
        """``visual`` is (B, C, H, W) float, as the wrappers produce it."""
        out = visual
        for stage in self.tower:
            out = stage["conv"](out)
            out = F.max_pool2d(out, 3, 2, padding=1)
            out = stage["block1"](out)
            out = stage["block2"](out)
        out = F.elu(out)
        # channels last before flattening: the order the dense layer's weights expect
        return out.permute(0, 2, 3, 1).reshape(out.shape[0], -1)

    def forward(
        self,
        visual: torch.Tensor,
        vels: torch.Tensor,
        state: torch.Tensor,
        dones: torch.Tensor,
        env_num: int,
    ) -> tuple:
        """The batch is environment-major: the entry for environment e at step t
        sits at ``e * steps_num + t``, which is what the recurrent unrolling and
        the sequence slicing both assume. Returns the ``(env_num * steps_num,
        LSTM_UNITS)`` recurrent output and the state carried out of the batch."""
        steps_num = visual.shape[0] // env_num
        flat = self.features(visual)
        hidden = torch.cat([F.elu(self.vels_hidden(vels)), F.elu(self.visual_hidden(flat))], dim=-1)
        hidden = F.elu(self.joint_hidden(hidden))

        sequence = hidden.reshape(env_num, steps_num, -1).unbind(dim=1)
        mask_sequence = dones.to(hidden.dtype).reshape(env_num, steps_num).unbind(dim=1)
        outputs, lstm_state = self.lstm(sequence, state, mask_sequence)

        return torch.stack(outputs, dim=1).reshape(-1, LSTM_UNITS), lstm_state


class AnimalPPONetwork(AnimalBackbone):
    """The backbone plus the two PPO heads: categorical logits and a state value."""

    def __init__(self, observation_space_shape: tuple[int, ...], vels_size: int) -> None:
        super().__init__(observation_space_shape, vels_size)
        self.value_head = nn.Linear(LSTM_UNITS, 1)
        self.logits_head = nn.Linear(LSTM_UNITS, ACTION_NUM)

    def forward(
        self,
        visual: torch.Tensor,
        vels: torch.Tensor,
        state: torch.Tensor,
        dones: torch.Tensor,
        env_num: int,
    ) -> tuple:
        lstm_out, lstm_state = super().forward(visual, vels, state, dones, env_num)
        return self.logits_head(lstm_out), self.value_head(lstm_out), lstm_state
