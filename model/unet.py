import math
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import asdict, dataclass
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

class FlexiblePositionalEncoding(nn.Module):
    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        assert d_model % 2 == 0
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # 保持原有的离散嵌入
        pe = self._create_pe_matrix(max_seq_len, d_model)
        self.register_buffer('pe', pe)
        
    def _create_pe_matrix(self, max_seq_len, d_model):
        i_seq = torch.linspace(0, max_seq_len - 1, max_seq_len)
        j_seq = torch.linspace(0, d_model - 2, d_model // 2)
        pos, two_i = torch.meshgrid(i_seq, j_seq)
        pe_2i = torch.sin(pos / 10000 ** (two_i / d_model))
        pe_2i_1 = torch.cos(pos / 10000 ** (two_i / d_model))
        return torch.stack((pe_2i, pe_2i_1), 2).reshape(max_seq_len, d_model)
    
    def forward(self, t):
        if t.dtype in [torch.int32, torch.int64, torch.long]:
            # 对整数时间步使用离散嵌入
            return self.pe[t]
        else:
            # 对浮点数时间步使用连续编码
            device = t.device
            half_dim = self.d_model // 2
            # embeddings = math.log(10000) / (half_dim - 1)
            embeddings = torch.log(torch.tensor(10000.)) / (half_dim - 1)
            embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
            embeddings = t[:, None] * embeddings[None, :]
            embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
            return embeddings


class UnetBlock(nn.Module):
    def __init__(self, shape, in_c, out_c, residual=False):
        """
        The only dimension that changes is the channel dimension (from in_c to out_c). The height and width remain the same because:

            Padding of 1 adds one pixel border around the image
            Kernel size of 3 and stride of 1 maintain spatial dimensions when combined with padding of 1
        """
        super().__init__()
        self.ln = nn.LayerNorm(shape)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.activation = nn.ReLU()
        self.residual = residual
        if residual:
            if in_c == out_c:
                self.residual_conv = nn.Identity()
            else:
                self.residual_conv = nn.Conv2d(in_c, out_c, 1)

    def forward(self, x):
        out = self.ln(x)
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        if self.residual:
            out += self.residual_conv(x)
        out = self.activation(out)
        return out
    


@dataclass
class UNetConfig:
    n_steps: int = 1000
    channels: List[int] = field(default_factory=lambda: [10, 20, 40, 80])
    pe_dim: int = 10
    residual: bool = False
    image_shape: Optional[Tuple[int, ...]] = None

class UNet(nn.Module):
    def __init__(self, config:UNetConfig):
        super().__init__()
        self.channels = config.channels
        if config.image_shape is None:
            C, H, W = (3, 28, 28)
        else:
            C, H, W = config.image_shape
            
        self.initial_channels = C

        # Calculate shapes for each level
        self.shapes = self._calculate_shapes(H, W)

        # Initialize components
        self.pe = self._build_positional_encoding(config.n_steps, config.pe_dim)
        self.encoder = self._build_encoder(config.pe_dim, config.residual)
        self.middle_block = self._build_middle_block(config.pe_dim, config.residual)
        self.decoder = self._build_decoder(config.pe_dim, config.residual)
        self.projector = self._build_projector()

    def _calculate_shapes(self, H, W):
        shapes = [(H, W)]
        cH, cW = H, W
        for _ in range(len(self.channels) - 1):
            cH //= 2
            cW //= 2
            shapes.append((cH, cW))
        return shapes

    def _build_positional_encoding(self, n_steps, pe_dim):
        return FlexiblePositionalEncoding(n_steps, pe_dim)

    def _build_encoder(self, pe_dim, residual):
        encoders = nn.ModuleList()
        pe_linears = nn.ModuleList()
        downs = nn.ModuleList()

        prev_channel = self.initial_channels
        for i, channel in enumerate(self.channels[:-1]):
            # PE linear projection
            pe_linears.append(
                nn.Sequential(
                    nn.Linear(pe_dim, prev_channel),
                    nn.ReLU(),
                    nn.Linear(prev_channel, prev_channel),
                )
            )

            # Double convolution block
            H, W = self.shapes[i]
            encoders.append(
                nn.Sequential(
                    UnetBlock(
                        (prev_channel, H, W), prev_channel, channel, residual=residual
                    ),
                    UnetBlock((channel, H, W), channel, channel, residual=residual),
                )
            )

            # Downsampling
            downs.append(nn.Conv2d(channel, channel, 2, 2))
            prev_channel = channel

        return nn.ModuleDict(
            {"encoders": encoders, "pe_linears": pe_linears, "downs": downs}
        )

    def _build_middle_block(self, pe_dim, residual):
        prev_channel = self.channels[-2]
        channel = self.channels[-1]
        H, W = self.shapes[-1]

        return nn.ModuleDict(
            {
                "pe_linear": nn.Linear(pe_dim, prev_channel),
                "block": nn.Sequential(
                    UnetBlock(
                        (prev_channel, H, W), prev_channel, channel, residual=residual
                    ),
                    UnetBlock((channel, H, W), channel, channel, residual=residual),
                ),
            }
        )

    def _build_decoder(self, pe_dim, residual):
        decoders = nn.ModuleList()
        pe_linears = nn.ModuleList()
        ups = nn.ModuleList()

        prev_channel = self.channels[-1]
        for channel, (H, W) in zip(self.channels[-2::-1], self.shapes[-2::-1]):
            pe_linears.append(nn.Linear(pe_dim, prev_channel))
            ups.append(nn.ConvTranspose2d(prev_channel, channel, 2, 2))

            decoders.append(
                nn.Sequential(
                    UnetBlock(
                        (channel * 2, H, W), channel * 2, channel, residual=residual
                    ),
                    UnetBlock((channel, H, W), channel, channel, residual=residual),
                )
            )
            prev_channel = channel

        return nn.ModuleDict(
            {"decoders": decoders, "pe_linears": pe_linears, "ups": ups}
        )

    def _build_projector(self):
        return nn.Conv2d(self.channels[0], self.initial_channels, 3, 1, 1)

    def encode(self, x, t):
        n = t.shape[0]
        t = self.pe(t)

        encoder_outs = []
        for pe_linear, encoder, down in zip(
            self.encoder["pe_linears"], self.encoder["encoders"], self.encoder["downs"]
        ):
            pe = pe_linear(t).reshape(n, -1, 1, 1)
            x = encoder(x + pe)
            encoder_outs.append(x)
            x = down(x)

        return x, encoder_outs

    def decode(self, x, encoder_outs, t):
        n = t.shape[0]

        for pe_linear, decoder, up, skip in zip(
            self.decoder["pe_linears"],
            self.decoder["decoders"],
            self.decoder["ups"],
            encoder_outs[::-1],
        ):
            pe = pe_linear(t).reshape(n, -1, 1, 1)
            x = up(x)

            # Handle padding if needed
            pad_x = skip.shape[2] - x.shape[2]
            pad_y = skip.shape[3] - x.shape[3]
            x = F.pad(
                x, (pad_x // 2, pad_x - pad_x // 2, pad_y // 2, pad_y - pad_y // 2)
            )

            x = torch.cat((skip, x), dim=1)
            x = decoder(x + pe)

        return x

    def forward(self, x, t):
        # Encoding path
        x, encoder_outs = self.encode(x, t)

        # Middle block
        n = t.shape[0]
        t = self.pe(t)
        pe_mid = self.middle_block["pe_linear"](t).reshape(n, -1, 1, 1)
        x = self.middle_block["block"](x + pe_mid)

        # Decoding path
        x = self.decode(x, encoder_outs, t)

        # Final projection
        x = self.projector(x)

        return x

    def save_pretrained(self, pretrained_path: str) -> None:
        os.makedirs(pretrained_path, exist_ok=True)

        with open(os.path.join(pretrained_path, "config.json"), mode="w") as f:
            json.dump(asdict(self.config), f)

        torch.save(self.state_dict(), os.path.join(pretrained_path, "model.pt"))

    @classmethod
    def from_pretrained(cls, pretrained_path: str) -> "UNet":
        with open(os.path.join(pretrained_path, "config.json"), mode="r") as f:
            config_dict = json.load(f)
        config = UNetConfig(**config_dict)

        model = cls(config)

        state_dict = torch.load(
            os.path.join(pretrained_path, "model.pt"), map_location=torch.device("cpu")
        )
        model.load_state_dict(state_dict)

        return model


unet_1_cfg = {"type": "UNet", "channels": [10, 20, 40, 80], "pe_dim": 128}
unet_res_cfg = {
    "type": "UNet",
    "channels": [10, 20, 40, 80],
    "pe_dim": 128,
    "residual": True,
}



