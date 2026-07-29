import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715))
                * torch.pow(x, 3)
            )
        )


# Next, to get an idea of what this GELU function looks
# like and how it compares to the
# ReLU function, let’s plot these functions side by side:

x = torch.linspace(-3, 3, 100)
gelu, relu = GELU(), nn.ReLU()

x = torch.linspace(-3, 3, 100)  # Creates 100 sample data points in the range –3 to 3
y_gelu, y_relu = gelu(x), relu(x)
plt.figure(figsize=(8, 3))
for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "RELU"]), 1):
    plt.subplot(1, 2, i)
    plt.plot(x, y)
    plt.title(f"{label} activation function ")
    plt.xlabel("x")
    plt.ylabel(f"{label} (x)")
    plt.grid(True)
plt.tight_layout()
plt.show()
