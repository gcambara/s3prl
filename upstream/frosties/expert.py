import torch
from torch.nn.utils.rnn import pad_sequence
from upstream.interfaces import UpstreamBase
from src.models.frosties import Frosties

class UpstreamExpert(UpstreamBase):
    def __init__(self, ckpt, model_config, **kwargs):
        super().__init__(**kwargs)

        cfg = torch.load(model_config)
        self.model = Frosties(cfg).load_from_checkpoint(ckpt)

    def forward(self, wavs):
        wavs = pad_sequence(wavs, batch_first=True)
        wavs = wavs.unsqueeze(1)

        features = self.model(wavs)  # (batch_size, feature_dim, extracted_seqlen)
        features = features.transpose(
            1, 2
        ).contiguous()  # (batch_size, extracted_seqlen, feature_dim)

        return {"default": features}