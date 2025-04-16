import torch
from torch import nn

import random

device = torch.accelerator.current_accelerator(
).type if torch.accelerator.is_available() else "cpu"

print(f"Using {device} device")


class AdderNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def decompose_number(n):
    ans = [0.0 for i in range(32)]
    for i in range(31):
        ans[i] = float(n & 1)
        n //= 2
    return ans


def bits_to_number(bits):
    num = 0
    for i in reversed(range(32)):
        num *= 2
        if bits[i] > 0.5:
            num += 1
    return num


def get_ans(bits):
    num_1 = bits_to_number(bits[0:32])
    num_2 = bits_to_number(bits[32:64])
    return decompose_number(num_1 + num_2)


print(get_ans(decompose_number(7) + decompose_number(8)))
