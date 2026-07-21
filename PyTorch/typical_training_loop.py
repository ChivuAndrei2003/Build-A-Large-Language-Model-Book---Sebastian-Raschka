from cProfile import label
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

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


train_loader = DataLoader(
    #
    # The ToyDataset instance created earlier serves asinput to the data loader.
    dataset=train_ds,
    batch_size=2,
    shuffle=True,  # Whether or not to shuffle the data
    num_workers=0,  # The number of background processes
    #
)
test_loader = DataLoader(
    #
    dataset=test_ds,
    batch_size=2,
    shuffle=False,
    num_workers=1,
)


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


torch.manual_seed(123)

# The dataset has two features and two classes
model = NeuralNetwork(num_inputs=2, num_outputs=2)

# The optimizer needs to know which parameters to optimize.
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
num_epochs = 3

for epoch in range(num_epochs):
    model.train()

    for batch_idx, (features, labels) in enumerate(train_loader):

        logits = model(features)

        loss = F.cross_entropy(logits, labels)  ## loss function

        # Sets the gradients from the previous round to 0 to
        # prevent unintended gradient accumulation
        optimizer.zero_grad()

        #   Computes the gradients of the loss given the model parameters
        loss.backward()

        # The optimizer uses the gradients to update the model parameters.
        optimizer.step()

        ### LOGGING
        print(
            f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
            f" | Batch {batch_idx+1:03d}/{len(train_loader):03d}"
            f" | Train/Val Loss: {loss:.2f}"
        )

        model.eval()
    # Optional model evaluation

# NOTE To prevent undesired gradient accumulation, it is important to include
# an optimizer.zero_grad() call in each update round to reset the gradients to
# 0. Otherwise, the gradients will accumulate, which may be undesired.

# After we have trained the model, we can use it to make predictions:
model.eval()
with torch.no_grad():
    outputs = model(X_train)
print(outputs)

# To obtain the class membership probabilities, we can then use PyTorch’s softmax
# function:
torch.set_printoptions(sci_mode=False)
probas = torch.softmax(outputs, dim=1)
print(probas)

# NOTE We can convert these values into class label predictions using PyTorch’s argmax
# function, which returns the index position of the highest value in each row if we set
# dim=1 (setting dim=0 would return the highest value in each column instead):
predictions = torch.argmax(probas, dim=1)
print(predictions)

# We can double-check this using the == comparison operator:
print(predictions == y_train)


# Since the dataset consists of five training examples, we have five out of five predictions
# that are correct, which has 5/5 × 100% = 100% prediction accuracy.
# To generalize the computation of the prediction accuracy, let’s implement a
# compute_accuracy function, as shown in the following listing.
def compute_accuracy(model, dataloader):

    model.eval()
    correct = 0.0
    total_examples = 0

    for idx, (features, labels) in enumerate(dataloader):

        with torch.no_grad():
            logits = model(features)

        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)

        return (correct / total_examples).item()


print("\n Compute accuracy : ", compute_accuracy(model, train_loader))
compute_accuracy(model, train_loader)
# compute_accuracy(model, test_loader, device=torch.device)
