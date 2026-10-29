import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet3D(nn.Module):
    """Encoder-decoder with skip connections. Output = raw logits (one channel per region)."""

    def __init__(self, in_ch=4, out_ch=3, base=16, depth=4):
        super().__init__()
        chs = [base * 2 ** i for i in range(depth + 1)]  # 16,32,64,128,256
        self.enc = nn.ModuleList([ConvBlock(in_ch, chs[0])] +
                                 [ConvBlock(chs[i - 1], chs[i]) for i in range(1, depth + 1)])
        self.pool = nn.MaxPool3d(2)
        self.up = nn.ModuleList([nn.ConvTranspose3d(chs[i], chs[i - 1], 2, stride=2)
                                 for i in range(depth, 0, -1)])
        self.dec = nn.ModuleList([ConvBlock(chs[i - 1] * 2, chs[i - 1])
                                  for i in range(depth, 0, -1)])
        self.head = nn.Conv3d(chs[0], out_ch, 1)

    def forward(self, x):
        skips = []
        for i, block in enumerate(self.enc):
            x = block(x if i == 0 else self.pool(x))
            skips.append(x)
        skips.pop()  # bottleneck output is x itself
        for up, dec in zip(self.up, self.dec):
            x = up(x)
            x = dec(torch.cat([x, skips.pop()], dim=1))
        return self.head(x)


if __name__ == "__main__":
    m = UNet3D()
    print(m(torch.randn(1, 4, 64, 64, 64)).shape)  # -> (1, 3, 64, 64, 64)
    print(sum(p.numel() for p in m.parameters()) / 1e6, "M params")
