# SPDX-License-Identifier: MIT
import numpy as np


class ActionTokenizer:
    """Maps continuous actions in [-1, 1] to discrete token IDs at the end of the vocabulary."""

    def __init__(self, vocab_size: int) -> None:
        self.n_bins = 128
        self.bins = np.linspace(-1.0, 1.0, self.n_bins + 1)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        self.action_token_begin_idx = vocab_size - self.n_bins

    def encode(self, action: np.ndarray) -> np.ndarray:
        """action: (...,) -> token_ids: (...,). Discretize and map to vocab tail."""
        discretized = np.digitize(action.clip(-1, 1), self.bins[1:-1])
        return self.action_token_begin_idx + discretized

    def decode(self, token_ids: np.ndarray) -> np.ndarray:
        """token_ids: (...,) -> action: (...,). Map token IDs back to continuous values."""
        bin_idx = np.clip(token_ids - self.action_token_begin_idx, 0, self.n_bins - 1)
        return self.bin_centers[bin_idx]


if __name__ == "__main__":
    tok = ActionTokenizer(151936)

    print(f"n_bins: {tok.n_bins}")
    print(f"action_token_begin_idx: {tok.action_token_begin_idx}")
    print(
        f"token ID range: [{tok.action_token_begin_idx}, {tok.action_token_begin_idx + tok.n_bins - 1}]"
    )
    print(f"bin width: {tok.bins[1] - tok.bins[0]:.6f}")
    print()

    # Encode/decode single values
    test_values = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    token_ids = tok.encode(test_values)
    reconstructed = tok.decode(token_ids)
    print("Single values:")
    for v, tid, r in zip(test_values, token_ids, reconstructed):
        print(f"  {v:+.4f} -> token {tid} -> {r:+.6f}  (error={abs(v - r):.6f})")
    print()

    # Roundtrip on action chunk: (horizon=4, action_dim=7)
    action_chunk = np.random.uniform(-1, 1, (4, 7))
    encoded = tok.encode(action_chunk)
    decoded = tok.decode(encoded)
    max_err = np.abs(action_chunk - decoded).max()
    print(f"Random chunk (4, 7): max roundtrip error = {max_err:.6f}")
    print(f"  encoded shape: {encoded.shape}, dtype: {encoded.dtype}")
    print(f"  decoded shape: {decoded.shape}, dtype: {decoded.dtype}")
    print()

    # Clipping behavior
    out_of_range = np.array([-1.5, 1.5])
    clipped_ids = tok.encode(out_of_range)
    clipped_vals = tok.decode(clipped_ids)
    print("Out-of-range clipping:")
    for v, tid, r in zip(out_of_range, clipped_ids, clipped_vals):
        print(f"  {v:+.4f} -> token {tid} -> {r:+.6f}")
