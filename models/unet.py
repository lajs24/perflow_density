"""2D U-Net backbone with time conditioning for PerFlow."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.embeddings import SinusoidalTimeEmbedding, TimeMLP


class ResidualBlock(nn.Module):
    """Conv residual block with time embedding conditioning."""

    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.norm2 = nn.GroupNorm(min(32, out_ch), out_ch)

        self.time_mlp = nn.Linear(time_emb_dim, out_ch * 2)
        self.silu = nn.SiLU()

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # First conv
        h = self.conv1(x)
        h = self.norm1(h)

        # Time conditioning: scale + shift
        time_params = self.time_mlp(self.silu(t_emb))  # (B, out_ch*2)
        scale, shift = time_params.chunk(2, dim=1)     # (B, out_ch), (B, out_ch)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        h = h * (1.0 + scale) + shift

        h = self.silu(h)

        # Second conv
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.silu(h)

        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Simple self-attention block."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.num_heads = num_heads
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_flat = x.reshape(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
        x_norm = self.norm(x_flat)

        qkv = self.qkv(x_norm).reshape(B, -1, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each (B, heads, H*W, head_dim)

        attn = (q @ k.transpose(-2, -1)) * (C // self.num_heads) ** -0.5
        attn = attn.softmax(dim=-1)

        h = (attn @ v).transpose(1, 2).reshape(B, H * W, C)
        h = self.proj(h)
        h = h.permute(0, 2, 1).reshape(B, C, H, W)
        return x + h


class DownBlock(nn.Module):
    """Encoder block: ResBlock + optional Attention + Downsample."""

    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, num_heads: int = 0):
        super().__init__()
        self.res = ResidualBlock(in_ch, out_ch, time_emb_dim)
        self.attn = AttentionBlock(out_ch, num_heads) if num_heads > 0 else nn.Identity()
        self.down = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> tuple:
        h = self.res(x, t_emb)
        h = self.attn(h)
        h_down = self.down(h)
        return h, h_down


class UpBlock(nn.Module):
    """Decoder block: Upsample → Concat → ResBlock + optional Attention."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, time_emb_dim: int, num_heads: int = 0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 4, stride=2, padding=1)
        self.res = ResidualBlock(in_ch + skip_ch, out_ch, time_emb_dim)
        self.attn = AttentionBlock(out_ch, num_heads) if num_heads > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle spatial size mismatch
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        h = torch.cat([x, skip], dim=1)
        h = self.res(h, t_emb)
        h = self.attn(h)
        return h


class UNet(nn.Module):
    """2D U-Net with time conditioning for Rectified Flow.

    Args:
        in_channels: 6 (x_t:2 + y_obs:2 + mask:2)
        out_channels: 2 (vx, vy)
        base_channels: 64
        channel_multipliers: [1, 2, 4, 8]
        num_res_blocks: 2
        num_heads: 4
    """

    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int = 2,
        base_channels: int = 64,
        channel_multipliers: list = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        num_heads: int = 4,
    ):
        super().__init__()

        # Time embedding
        time_emb_dim = base_channels * 4
        self.time_embed = SinusoidalTimeEmbedding(base_channels)
        self.time_mlp = TimeMLP(base_channels, time_emb_dim)

        # Initial projection
        self.proj_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Encoder
        self.enc_blocks = nn.ModuleList()
        ch = base_channels
        skip_channels = []
        for mult in channel_multipliers:
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.enc_blocks.append(DownBlock(ch, out_ch, time_emb_dim, num_heads))
                skip_channels.append(out_ch)
                ch = out_ch

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock(ch, ch, time_emb_dim),
            AttentionBlock(ch, num_heads),
            ResidualBlock(ch, ch, time_emb_dim),
        )

        # Decoder
        self.dec_blocks = nn.ModuleList()
        ch = base_channels * channel_multipliers[-1]
        for mult in reversed(channel_multipliers):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                skip_ch = skip_channels.pop()
                self.dec_blocks.append(UpBlock(ch, skip_ch, out_ch, time_emb_dim, num_heads))
                ch = out_ch

        # Output projection
        self.proj_out = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.GroupNorm(min(32, base_channels), base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) conditioned input (x_t, y_obs, mask concatenated)
            t: (B,) time steps in [0, 1]

        Returns:
            v: (B, out_channels, H, W) predicted velocity
        """
        # Time embedding
        t_emb = self.time_embed(t)
        t_emb = self.time_mlp(t_emb)

        # Initial projection
        h = self.proj_in(x)

        # Encoder
        skips = []
        for block in self.enc_blocks:
            skip, h = block(h, t_emb)
            skips.append(skip)

        # Bottleneck
        h = self.bottleneck[0](h, t_emb)
        h = self.bottleneck[1](h)
        h = self.bottleneck[2](h, t_emb)

        # Decoder
        for block in self.dec_blocks:
            skip = skips.pop()
            h = block(h, skip, t_emb)

        return self.proj_out(h)
