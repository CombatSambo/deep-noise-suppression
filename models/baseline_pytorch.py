from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class BaselineModelConfig:
    input_size: int = 322
    hidden_size: int = 322
    output_size: int = 161


class BaselineGRUMaskNet(nn.Module):
    """
    PyTorch re-implementation of the recurrent masking baseline used by the ONNX model.

    Notes:
    - IO signatures are intentionally aligned with baseline ONNX inference:
      input: [seq_len, batch_size, 322], h01: [1, batch_size, 322], h02: [1, batch_size, 322]
      output mask: [seq_len, batch_size, 161], hn1: [1, batch_size, 322], hn2: [1, batch_size, 322]
    - Topology parity with Netron graph:
      GRU_0(input, h01) -> Add(input, squeeze(GRU_0_out)) -> GRU_3(..., h02) -> Linear -> Sigmoid -> Clip(min=0)
    - TODO: map ONNX initializers to this module when ONNX weights export is available in this environment.
    - TODO: exact GRU parity may still differ because ONNX graph uses linear_before_reset=1.
    """

    def __init__(self, config: Optional[BaselineModelConfig] = None) -> None:
        super().__init__()
        self.config = config or BaselineModelConfig()

        self.gru1 = nn.GRU(
            input_size=self.config.input_size,
            hidden_size=self.config.hidden_size,
            num_layers=1,
            batch_first=False,
        )
        self.gru2 = nn.GRU(
            input_size=self.config.input_size,
            hidden_size=self.config.hidden_size,
            num_layers=1,
            batch_first=False,
        )
        self.mask_proj = nn.Linear(self.config.hidden_size, self.config.output_size)

    def init_hidden(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h01 = torch.zeros((1, batch_size, self.config.hidden_size), device=device, dtype=dtype)
        h02 = torch.zeros((1, batch_size, self.config.hidden_size), device=device, dtype=dtype)
        return h01, h02

    def forward(
        self,
        x: torch.Tensor,
        h01: Optional[torch.Tensor] = None,
        h02: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape [seq_len, batch_size, feat_dim], got {tuple(x.shape)}")
        if x.shape[-1] != self.config.input_size:
            raise ValueError(
                f"Expected input feature size {self.config.input_size}, got {x.shape[-1]}"
            )

        batch_size = x.shape[1]
        if h01 is None or h02 is None:
            h01, h02 = self.init_hidden(batch_size=batch_size, device=x.device, dtype=x.dtype)

        out1, hn1 = self.gru1(x, h01.contiguous())
        residual = x + out1
        out2, hn2 = self.gru2(residual, h02.contiguous())
        mask = torch.sigmoid(self.mask_proj(out2))
        mask = torch.clamp(mask, min=0.0)
        return mask, hn1, hn2


class EncoderDecoderMaskNet(nn.Module):
    """
    Recurrent encoder-decoder style mask estimator.

    The module keeps the same IO contract as BaselineGRUMaskNet:
    - input: [seq_len, batch_size, 322]
    - output: mask [seq_len, batch_size, 161], hn1, hn2

    hn1/hn2 correspond to encoder/decoder GRU hidden states.
    """

    def __init__(self, config: Optional[BaselineModelConfig] = None) -> None:
        super().__init__()
        self.config = config or BaselineModelConfig()
        self.encoder_hidden_size = 256
        self.decoder_hidden_size = self.config.hidden_size

        self.encoder_gru = nn.GRU(
            input_size=self.config.input_size,
            hidden_size=self.encoder_hidden_size,
            num_layers=1,
            batch_first=False,
        )
        self.decoder_gru = nn.GRU(
            input_size=self.encoder_hidden_size,
            hidden_size=self.decoder_hidden_size,
            num_layers=1,
            batch_first=False,
        )
        self.mask_proj = nn.Linear(self.decoder_hidden_size, self.config.output_size)

    def init_hidden(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h01 = torch.zeros((1, batch_size, self.encoder_hidden_size), device=device, dtype=dtype)
        h02 = torch.zeros((1, batch_size, self.decoder_hidden_size), device=device, dtype=dtype)
        return h01, h02

    def forward(
        self,
        x: torch.Tensor,
        h01: Optional[torch.Tensor] = None,
        h02: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape [seq_len, batch_size, feat_dim], got {tuple(x.shape)}")
        if x.shape[-1] != self.config.input_size:
            raise ValueError(
                f"Expected input feature size {self.config.input_size}, got {x.shape[-1]}"
            )

        batch_size = x.shape[1]
        if h01 is None or h02 is None:
            h01, h02 = self.init_hidden(batch_size=batch_size, device=x.device, dtype=x.dtype)

        enc_out, hn1 = self.encoder_gru(x, h01.contiguous())
        dec_out, hn2 = self.decoder_gru(enc_out, h02.contiguous())
        mask = torch.sigmoid(self.mask_proj(dec_out))
        mask = torch.clamp(mask, min=0.0)
        return mask, hn1, hn2


def create_mask_model(model_type: str) -> nn.Module:
    model_key = model_type.strip().lower()
    if model_key == "gru":
        return BaselineGRUMaskNet()
    if model_key in {"encoder_decoder", "encdec", "ae"}:
        return EncoderDecoderMaskNet()
    raise ValueError(f"Unsupported model_type '{model_type}'. Use 'gru' or 'encoder_decoder'.")
