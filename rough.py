import torch

a = torch.tensor([1,2,3])

print(a)
print("Is GPU AVAIL: " ,torch.cuda.is_available())