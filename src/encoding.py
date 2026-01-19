import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, num_freqs, include_input=True, log_sampling=True):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.log_sampling = log_sampling
        self.funcs = [torch.sin, torch.cos]
        self.out_dim = 0
        if include_input:
            self.out_dim += 3
        self.out_dim += 3 * 2 * num_freqs

        if log_sampling:
            self.freq_bands = 2.0 ** torch.linspace(0.0, num_freqs - 1, num_freqs)
        else:
            self.freq_bands = torch.linspace(1.0, 2.0**(num_freqs - 1), num_freqs)

    def forward(self, x):
        embed = [x] if self.include_input else []
        for freq in self.freq_bands:
            for func in self.funcs:
                embed.append(func(x * freq))
        return torch.cat(embed, dim=-1)
