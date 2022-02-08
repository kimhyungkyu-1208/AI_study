import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# 재사용을 위해 랜덤값을 초기화 합니다.
torch.manual_seed(1)

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[3], [6], [9]])

