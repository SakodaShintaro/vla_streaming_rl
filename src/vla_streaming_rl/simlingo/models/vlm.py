from torch import nn

from vla_streaming_rl.simlingo.models.encoder.internvl2_model import (
    LingoInternVLModel,
)


class VLMEncoderModel(nn.Module):
    def __init__(
        self,
        cfg_data_module,
        processor,
        **cfg,
    ):
        super().__init__()

        for key, value in cfg.items():
            setattr(self, key, value)
        for key, value in cfg_data_module.items():
            setattr(self, key, value)

        self.token_size = self.embed_dim

        if "internvl2" in self.variant.lower():
            self.image_encoder = LingoInternVLModel(self.variant)
        else:
            raise ValueError(f"Unknown variant {self.variant}")

        self.image_encoder.processor = processor
        self.image_encoder.use_global_img = self.use_global_img

        self.image_encoder.language_model = None
        self.image_encoder.model.language_model = None

        # freeze the parameters -> no gradient updates
        if self.freeze:
            # for p in self.image_encoder.model.base_model.parameters():
            for p in self.parameters():
                p.requires_grad = False
                # p.requires_grad = False

            for p in self.image_encoder.model.mlp1.parameters():
                p.requires_grad = True
