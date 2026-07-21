from ast import mod
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sympy import true
import torch
import torch.nn.functional as F
from torch.autograd import grad

y = torch.tensor([1.0])
x1 = torch.tensor([1.1])
w1 = torch.tensor([2.2], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)

z = x1 * w1 + b
a = torch.sigmoid(z)

loss = F.binary_cross_entropy(a, y)

grad_L_w1 = grad(loss, w1, retain_graph=True)

grad_L_b = grad(loss, b, retain_graph=True)

print()
print(grad_L_w1)
print(grad_L_b)

# EQUIVALENT :
print()
loss.backward()
print(w1.grad)
print(b.grad)
print()
# A.5
# Implementing multilayer neural networks


class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.layers = torch.nn.Sequential(
            #
            # 1st hidden layer
            # The Linear layer takes the number of input and output nodes as arguments
            torch.nn.Linear(num_inputs, 30),
            # Nonlinear activation functions are placed between the hidden layers.
            torch.nn.ReLU(),
            #
            # 2nd hidden layer
            # The number of output nodes of one hidden layer has to match the number
            # of inputs of the next layer.
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),
            #
            # output layer
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits  # the outputs of the last layer are called logits


model = NeuralNetwork(50, 3)
print(model)  # summary of its strcture

# Next, let’s check the total number of trainable parameters of this model:

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("\nTotal number of trainable model parameters:", num_params)

print("\nFirst linear layer :", model.layers[0].weight)

print("\nNumber of dimensions for first layer : ", model.layers[0].weight.shape)

print("\nBias vector for first layer : ", model.layers[0].bias)


torch.manual_seed(123)
model = NeuralNetwork(50, 3)
print("\n", model.layers[0].weight)

# let’s see how the NeuralNetwork instance is used via the forward pass:
torch.manual_seed(123)
X = torch.rand((1, 50))
out = model(X)
print(out)

# The forward pass refers to calculating output tensors from input tensors. This
# involves passing the input data through all the neural network layers, starting from
# the input layer, through hidden layers, and finally to the output layer.

#  The <AddmmBackward0> part of grad_fn=<AddmmBackward0> specifies the
# operation performed. In this case, it is an Addmm operation. Addmm stands for matrix
# multiplication (mm) followed by an addition (Add).

with torch.no_grad():
    out = model(X)
print(out)


# If we want to compute class-membership probabilities
# for our predictions, we have to call the softmax function explicitly:

with torch.no_grad():
    out = torch.softmax(model(X), dim=1)
print(out)

# Listing A.5 Creating a small toy dataset

X_train = torch.tensor(
    [[-1.2, 3.1], [-0.9, 2.9], [-0.5, 2.6], [2.3, -1.1], [2.7, -1.5]]
)
y_train = torch.tensor([0, 0, 0, 1, 1])
X_test = torch.tensor(
    [
        [-0.8, 2.8],
        [2.6, -1.6],
    ]
)
y_test = torch.tensor([0, 1])

# Listing A.6 Defining a custom Dataset class

from torch.utils.data import Dataset


# Listing A.6 Defining a custom Dataset class
class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        one_x = self.features[index]  # Instructions for retrieving
        one_y = self.labels[index]  # exactly one data record and
        return one_x, one_y  # the corresponding label

    # Instructions for
    # returning the total
    # length of the dataset :
    def __len__(self):
        return self.labels.shape[0]


train_ds = ToyDataset(X_train, y_train)
test_ds = ToyDataset(X_test, y_test)
print("\n", len(train_ds))  # 5


# Listing A.7 Instantiating data loaders
torch.manual_seed(123)

train_loader = DataLoader(
    #
    # The ToyDataset instance created earlier serves asinput to the data loader.
    dataset=train_ds,
    batch_size=2,
    shuffle=True,  # Whether or not to shuffle the data
    num_workers=0,  # The number of background processes
    #
)
test_ds = ToyDataset(X_test, y_test)

test_loader = DataLoader(
    dataset=test_ds,
    batch_size=2,
    shuffle=False,  # not necessary to shuffle a dataset
    num_workers=0,
)

# After instantiating the training data loader, we can iterate over it.
# The iteration over the test_loader works similarly but is omitted for brevity:

for idx, (x, y) in enumerate(train_loader):
    print(f"\n Batch {idx+1}: ", x, y)

# We've update cycles during training.
# We specified a batch size of 2 here, but the third batch only contains
# a single example.
# That’s because we have five training examples, and 5 is not evenly divisible by 2.

# In practice, having a substantially smaller batch as the last batch in a training epoch
# can disturb the convergence during training. To prevent this, set drop_last=True,
# which will drop the last batch in each epoch, as shown in the following listing.


# Listing A.8 A training loader that drops the last batch
train_loader = DataLoader(
    #
    dataset=train_ds,
    batch_size=2,
    shuffle=True,
    num_workers=0,
    drop_last=True,
)

# Now, iterating over the training loader, we can see that the last batch is omitted:
for idx, (x, y) in enumerate(train_loader):
    print(f"\nBatch {idx+1}:", x, y)
