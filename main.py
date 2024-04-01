import torch
from utils import iou

t1 = torch.tensor([1.0, 1.0, 2.0, 2.0]).unsqueeze(0)
t2 = torch.tensor([2.0, 2.0, 3.0, 2.0]).unsqueeze(0)
print(iou(t1, t2))

