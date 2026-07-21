import torch
import torch.nn.functional as F

print(torch.__version__)
print(torch.backends.mps.is_available())

# Creating PyTorch tensors

# Example 1 : Creates a zero-dimensional tensor (scalar) from a Python integer
tensor0d = torch.tensor(1)

# Example 2 : Creates a one-dimensional tensor (vector) from a Python list
tensor1d = torch.tensor([1, 2, 3])

# Example 3 :
tensor2d = torch.tensor(
    [[1, 2], [3, 4]]  # Creates a two-dimensional tensor
)  # from a nested Python list

# Example 4 :Creates a three-dimensional
# tensor from a nested Python list :

tensor3d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

# Tensor data types

## access the data type of a tensor :
tensor1d = torch.tensor([1, 2, 3])
print(tensor1d.dtype)
print()
# tensor frm floats
floatvec = torch.tensor([1.0, 2.0, 3.0])
print(floatvec.dtype)
print("\n Changing data type from int to float to reduce computations : ")
floatvec = tensor1d.to(torch.float32)
print(floatvec)
print(floatvec.dtype)

print()
# Common PyTorch tensor operations

tensor2d = torch.tensor([[1, 2, 3], [4, 5, 6]])
print("\nTensor 2D  :", tensor2d)
print("\nTensor 2D shape :", tensor2d.shape)

print("\nTo reshape the tensor into a 3 × 2 tensor, we can use the .reshape method:")
print(tensor2d.reshape(3, 2))
print()
print("""\nHowever, note that the more common command for 
      reshaping tensors in PyTorch is .view():""")
print(tensor2d.view(3, 2))

print("""\nNext, we can use .T to transpose a tensor, 
      which means flipping it across its diagonal.
Note that this is similar to reshaping a tensor, 
as you can see based on the follow-
ing result:\n""")

print(tensor2d.T)

print(
    "\nLastly, the common way to multiply two matrices in PyTorch is the .matmul method:"
)
print(tensor2d.matmul(tensor2d.T))

print(
    """\nHowever, we can also adopt the @ operator, which accomplishes the same thing more
compactly:"""
)

print(tensor2d @ tensor2d.T)


# Seeing models as computation graphs


# The code in the following listing implements the forward pass (prediction step) of a
# simple logistic regression classifier, which can be seen as a single-layer neural network.
# It returns a score between 0 and 1, which is compared to the true class label (0 or 1)
# when computing the loss.


print("\nA logistic regression forward pass : ")
y = torch.tensor([1.0])  # true label
x1 = torch.tensor([1.1])  # input feature
w1 = torch.tensor([2.2])  # weight parameter
b = torch.tensor([0.0])  # bias unit

z = x1 * w1 + b  # net input

a = torch.sigmoid(z)  # activation & output

loss = F.binary_cross_entropy(a, y)
print(loss)

# am ramas la pagina 263 apendix A si la pagina 40




