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


def gen_case():
    num_1 = random.randint(0, (1 << 31) - 1)
    num_2 = random.randint(0, (1 << 31) - 1)
    return decompose_number(num_1) + decompose_number(num_2)


def gen_data(n):
    return [gen_case() for i in range(n)]


def get_expected_output(data):
    [get_ans(bits) for bits in data]


model = AdderNetwork().to(device)
data = torch.tensor(gen_data(3), dtype=torch.float64)
results = model(data)
print(results)
