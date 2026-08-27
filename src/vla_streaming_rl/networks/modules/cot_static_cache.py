# SPDX-License-Identifier: MIT
"""A Qwen3.5 cache whose tensors never move, so a decode step can be graphed.

``Qwen3_5DynamicCache`` reassigns the linear-attention states to fresh tensors
every step and grows the attention keys and values with ``torch.cat``. A CUDA
graph records kernels together with the addresses they read and write, so a
cache that reallocates makes a recorded step replay against memory that is no
longer there. Everything here exists to hold the addresses still:

- the linear-attention states are copied into the tensor already in place, so
  the model's ``cache.conv_states[i] = state`` keeps working unchanged;
- the attention keys and values are one buffer per layer, written at the step's
  position, and handed back whole. ``is_compileable`` tells the mask builder not
  to take the ``is_causal`` shortcut, so causality masks the not-yet-written
  tail rather than the buffer's length doing it.

Buffers are shaped from the first tensors the model passes in, so no layer
geometry is duplicated from the config. The class is not a subclass: the cache
is duck-typed by the model, and inheriting would carry over exactly the
reallocating methods being replaced.
"""

import torch


class _InPlaceStates:
    """Per-layer states that keep their storage when the model assigns to them.

    ``cache.conv_states[i] = state`` on a plain list rebinds to whatever the
    layer just allocated, which moves the address every step. Copying into the
    tensor already there keeps it, and the model's code does not change.
    """

    def __init__(self, num_layers: int) -> None:
        self._states = [None] * num_layers
        self.written = [False] * num_layers

    def __len__(self) -> int:
        return len(self._states)

    def __getitem__(self, index: int):
        return self._states[index]

    def __setitem__(self, index: int, value: torch.Tensor) -> None:
        if self._states[index] is None:
            self._states[index] = torch.empty_like(value)
        self._states[index].copy_(value)
        self.written[index] = True

    def clear_written(self) -> None:
        self.written = [False] * len(self._states)


class CoTStaticCache:
    """The cache the chain carries. One instance lives for the whole run: a new
    chain resets it rather than replacing it, so the addresses a graph recorded
    stay valid across restarts."""

    # Read by the mask builder, which skips building an explicit mask and leans
    # on sdpa's is_causal when this is false. That shortcut assumes the keys and
    # values end where the sequence ends, which is exactly what a preallocated
    # buffer breaks.
    is_compileable = True

    def __init__(self, config, max_len: int) -> None:
        num_layers = config.num_hidden_layers
        self.layer_types = config.layer_types
        self.transformer_layers = [
            i for i in range(num_layers) if self.layer_types[i] == "full_attention"
        ]
        self.last_linear_layer = (
            len(self.layer_types) - 1 - self.layer_types[::-1].index("linear_attention")
        )
        self.max_len = max_len
        self.conv_states = _InPlaceStates(num_layers)
        self.recurrent_states = _InPlaceStates(num_layers)
        self.key_cache = [None] * num_layers
        self.value_cache = [None] * num_layers
        self.length = 0

    def __len__(self) -> int:
        return len(self.layer_types)

    def reset(self) -> None:
        """Start a new chain in the same memory. The stale contents are not
        cleared: the linear-attention states are written before they are read
        once ``has_previous_state`` is false again, and the attention tail is
        masked off by causality."""
        self.conv_states.clear_written()
        self.recurrent_states.clear_written()
        self.length = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache_position = cache_kwargs["cache_position"]
        keys = self._buffer(self.key_cache, layer_idx, key_states)
        values = self._buffer(self.value_cache, layer_idx, value_states)
        keys.index_copy_(2, cache_position, key_states)
        values.index_copy_(2, cache_position, value_states)
        return keys, values

    # ``get_seq_length`` and ``get_mask_sizes`` are the model's interface, called
    # with no layer index as often as with one, so the default stays.
    def get_seq_length(self, layer_idx: int | None = 0) -> int:
        del layer_idx
        return self.length

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
        """The whole buffer, always. A mask whose width tracked the filled length
        would change shape every step, which is what the graph cannot have."""
        del cache_position, layer_idx
        return self.max_len, 0

    @property
    def has_previous_state(self) -> bool:
        """False until a chain has written its last linear layer, which is what
        the dynamic cache signals by that state still being None."""
        return self.conv_states.written[self.last_linear_layer]

    def _buffer(self, cache: list, layer_idx: int, reference: torch.Tensor) -> torch.Tensor:
        if cache[layer_idx] is None:
            batch, heads, _, head_dim = reference.shape
            cache[layer_idx] = torch.zeros(
                (batch, heads, self.max_len, head_dim),
                dtype=reference.dtype,
                device=reference.device,
            )
        return cache[layer_idx]
