import os
from typing import Tuple, List

import torch
from torch import Tensor

from utility.download import _urls_to_filepaths
from .expert import UpstreamExpert as _UpstreamExpert


def frosties_local(ckpt, model_config, **kwargs):
    """
    The model from local ckpt
        ckpt (str): PATH
        model_config (str): PATH
    """
    assert os.path.isfile(ckpt)
    assert os.path.isfile(model_config)

    def hook_postprocess(hiddens: List[Tuple[str, Tensor]]):
        remained_hiddens = [x for x in hiddens if x[0] != "self.model"]
        final_hidden = [x for x in hiddens if x[0] == "self.model"]
        assert len(final_hidden) == 1
        final_hidden = final_hidden[0]

        updated_hiddens = []
        for identifier, tensor in remained_hiddens:
            updated_hiddens.append(
                (identifier, tensor.transpose(1, 2))
            )

        updated_hiddens.append((final_hidden[0], final_hidden[1].transpose(1, 2)))
        return updated_hiddens

    hooks = [
        (f"self.model", lambda input, output: output)
    ]

    # Read config
    cfg = torch.load(model_config)

    # Read the number of convolutional and transformer layers
    conv_layers, transformer_layers = len(cfg.conv_filters.split(',')), len(cfg.trn_dim.split(','))

    # Add conv skip connections
    if cfg.enable_conv_skip_connections:
        for i in range(conv_layers):
            hooks.append((f"self.model.encoder.local_encoder.conv_skip_connections[{i}]", lambda input, output: output))

    # Add transformer skip connections
    if cfg.enable_rnn_skip_connections:
        for i in range(transformer_layers - 1):
            hooks.append((f"self.model.encoder.context_encoder.trn_skip_connections[{i}]", lambda input, output: output))

    # Add transformer layers
    for i in range(transformer_layers - 1):
        hooks.append((f"self.model.encoder.context_encoder.context_encoder[{i}]", lambda input, output: output))

    kwargs["hooks"] = hooks
    kwargs["hook_postprocess"] = hook_postprocess

    return _UpstreamExpert(ckpt, model_config, **kwargs)