from typing import Any

import torch
import torch.nn as nn

# 4.5 A neural network to illustrate shortcut connections


class ExampleDeepNeuralNetwork(nn.Module):

    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), nn.GELU()),
                nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), nn.GELU()),
                nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), nn.GELU()),
                nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), nn.GELU()),
                nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), nn.GELU()),
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)  # Compute the output of the current layer

            # check if shortcut can be applied:
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x


# Let’s use this code to initialize a neural network without shortcut connections.
# Each layer will be initialized such that it accepts
# an example with three input values and returns three output values.
# The last layer returns a single output value:

layer_sizes = [3, 3, 3, 3, 3, 1]
sample_input = torch.tensor([[1.0, 0.0, -1.0]])
torch.manual_seed(123)
model_without_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=False)


# Next, we implement a function that computes the gradients in the model’s back-
# ward pass:
def print_gradients(model, x):
    # Forward pass
    output = model(x)
    target = torch.tensor([[0.0]])

    # Calculate loss based on how close the target
    # and output are
    loss = nn.MSELoss()
    loss = loss(output, target)

    # Backward pass to calculate the gradients
    loss.backward()

    for name, param in model.named_parameters():
        if "weight" in name:
            # Print the mean absolute gradient of the weights
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")


print()
print_gradients(
    model_without_shortcut, sample_input
)  # prints gradients thtt become smaller => vanishing gradient problem

# Let’s now instantiate a model with skip connections and see how it compares:
print()
torch.manual_seed(123)
model_with_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True)
print_gradients(model_with_shortcut, sample_input)
