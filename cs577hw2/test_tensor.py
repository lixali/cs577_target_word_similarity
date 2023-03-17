import torch

# Create a tensor
x = torch.tensor([[1.2, 2.2, 3.2], [4.2, 5.2, 6.2]])

# Calculate the mean of the tensor along the first dimension
mean = torch.mean(x, dim=0)

# Print the mean
print(mean)