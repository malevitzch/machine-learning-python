import torch
from torch import nn

import time
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")


class AdderNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def decompose_number(n):
    ans = [0.0 for i in range(8)]
    for i in range(8):
        ans[i] = float(n & 1)
        n //= 2
    return ans


def bits_to_number(bits):
    num = 0
    for i in reversed(range(len(bits))):
        num *= 2
        if bits[i] > 0.5:
            num += 1
    return num


def get_ans(bits):
    num_1 = bits_to_number(bits[0:8])
    num_2 = bits_to_number(bits[8:16])
    return decompose_number(num_1 + num_2)


def gen_case():
    num_1 = random.randint(0, (1 << 7) - 1)
    num_2 = random.randint(0, (1 << 7) - 1)
    return decompose_number(num_1) + decompose_number(num_2)


def gen_data(n):
    return [gen_case() for i in range(n)]


def get_expected_output(data):
    return [get_ans(bits) for bits in data]


model = AdderNetwork().to(device)

loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

start_time = time.time()
iters = 5000
for i in range(iters):
    model.train()
    inputs = gen_data(512)
    outputs = model(torch.tensor(inputs, dtype=torch.float32).to(device))
    optimizer.zero_grad()
    loss_val = loss(outputs, torch.tensor(
        get_expected_output(inputs), dtype=torch.float32).to(device))
    loss_val.backward()
    optimizer.step()

    if (i + 1) % 100 == 0:
        print(f"{i+1}th done")

end_time = time.time()
print(f"Training completed in {(end_time - start_time):.2f} seconds")

ans = input("Do you want to run the accuracy test? [y/n]\n")
if ans == "y":
    total = 128 * 128
    iters = 0
    ans = 0
    mistakes = []
    for i in range(128):
        for j in range(128):
            input_val = torch.tensor(
                [decompose_number(i) + decompose_number(j)]).to(device)
            output = model(input_val).detach().cpu().numpy()[0]
            if bits_to_number(output) == i + j:
                ans += 1
            else:
                mistakes.append((i, j, bits_to_number(output)))
            iters += 1
            if (iters + 1) % 100 == 0:
                print(f"{iters+1}/{total} done")
    percentage = 100 * (ans / total)
    print(f"Accuracy test finished: {percentage:0.2f}")

ans = input("Do you want to see where the model made mistakes? [y/n]\n")
if ans == "y":
    for (a, b, c) in mistakes:
        print(f"{a} + {b} =/= {c}")

ans = input("Do you want to enter interactive mode? [y/n]\n")
if ans == "y":
    while True:
        try:
            a = int(input("Enter first number: "))
            b = int(input("Enter second number: "))
        except ValueError:
            print("Invalid input\n")
            continue
        if a == -1 or b == -1:
            break
        input_vals = torch.tensor([decompose_number(a) + decompose_number(b)])
        output = model(input_vals).detach().numpy()[0]
        print(bits_to_number(output))
