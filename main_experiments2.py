import torch


tensor = torch.zeros([8, 8])

tensor[3, 5] = 1
print(
    torch.unravel_index(tensor.argmax(), tensor.shape)
)

#  I expect indices to be 4 5 0 4

flipped = tensor.flip([0])
print(
    torch.unravel_index(flipped.argmax(), flipped.shape)
)