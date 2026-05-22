"""Vendored OpenGVLab/InternVL2-1B trust_remote_code modeling.

Originally fetched by `AutoModel.from_pretrained("OpenGVLab/InternVL2-1B",
trust_remote_code=True)` from
  https://huggingface.co/OpenGVLab/InternVL2-1B
  revision 0d75ccd166b1d0b79446ae6c5d1a4a667f1e6187
Licensed under MIT (see file headers).

Vendored so we don't depend on transformers 5.x being able to load InternVL2's
HF-hosted modeling code unaltered (transformers 5.x wraps model __init__ in a
meta-device context and requires the new `all_tied_weights_keys` attribute set
by `post_init()`; the upstream modeling code predates both).

Look for `simlingo-vendor-fix:` markers to see what's changed.
"""
