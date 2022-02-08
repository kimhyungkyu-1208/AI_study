# 파이토치(PyTorch)에는 데이터 작업을 위한 기본 요소 두가지인 
# torch.utils.data.DataLoader 와 
# torch.utils.data.Dataset 가 있습니다. 
# Dataset 은 샘플과 정답(label)을 저장하고, 
# DataLoader 는 Dataset 을 순회 가능한 객체(iterable)로 감쌉니다.

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt

# 데이터 샘플을 처리하는 코드는 지저분(messy)하고 유지보수가 어려울 수 있습니다; 
# 더 나은 가독성(readability)과 모듈성(modularity)을 위해 
# 데이터셋 코드를 모델 학습 코드로부터 분리하는 것이 이상적입니다. 

# PyTorch는 torch.utils.data.DataLoader 와 
# torch.utils.data.Dataset 의 
# 두 가지 데이터 기본 요소를 제공하여 
# 미리 준비해된(pre-loaded) 데이터셋 뿐만 아니라 
# 가지고 있는 데이터를 사용할 수 있도록 합니다. 
# Dataset 은 샘플과 정답(label)을 저장하고, 
# DataLoader 는 Dataset 을 샘플에 쉽게 접근할 수 있도록 순회 가능한 객체(iterable)로 감쌉니다.

# PyTorch의 도메인 특화 라이브러리들은 (FashionMNIST와 같은) 
# 다양한 미리 준비해둔(pre-loaded) 데이터셋을 제공합니다. 
# 데이터셋은 torch.utils.data.Dataset 의 
# 하위 클래스로 개별 데이터를 특정하는 함수가 구현되어 있습니다. 
# 이러한 데이터셋은 모델을 만들어보고(prototype) 
# 성능을 측정(benchmark)하는데 사용할 수 있습니다. 
# 여기에서 데이터셋들을 찾아볼 수 있습니다: 
#   이미지 데이터셋, 텍스트 데이터셋 및 오디오 데이터셋

# 데이터셋 불러오기
# TorchVision 에서 Fashion-MNIST 데이터셋을 불러오는 예제를 살펴보겠습니다. 
# Fashion-MNIST는 Zalando의 기사 이미지 데이터셋으로 60,000개의 학습 예제와 
# 10,000개의 테스트 예제로 이루어져 있습니다. 
# 각 예제는 흑백(grayscale)의 28x28 이미지와 
# 10개 분류(class) 중 하나인 정답(label)으로 구성됩니다.

# 다음 매개변수들을 사용하여 FashionMNIST 데이터셋 을 불러옵니다:
# root 는 학습/테스트 데이터가 저장되는 경로입니다.
# train 은 학습용 또는 테스트용 데이터셋 여부를 지정합니다.
# download=True 는 root 에 데이터가 없는 경우 인터넷에서 다운로드합니다.
# transform 과 target_transform 은 특징(feature)과 정답(label) 변형(transform)을 지정합니다.
# 공개 데이터셋에서 학습 데이터를 내려받습니다.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# PyTorch는 TorchText, TorchVision 및 TorchAudio 와 같이 
# 도메인 특화 라이브러리를 데이터셋과 함께 제공하고 있습니다. 
# 이 튜토리얼에서는 TorchVision 데이터셋을 사용하도록 하겠습니다.

# torchvision.datasets 모듈은 
# CIFAR, COCO 등과 
# 같은 다양한 실제 비전(vision) 데이터에 대한 Dataset(전체 목록은 여기)을 포함하고 있습니다. 
# 이 튜토리얼에서는 FasionMNIST 데이터셋을 사용합니다. 
# 모든 TorchVision Dataset 은 샘플과 정답을 각각 변경하기 위한 
# transform 과 target_transform 의 두 인자를 포함합니다.

# 공개 데이터셋에서 테스트 데이터를 내려받습니다.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# 데이터셋을 순회하고 시각화하기
# Dataset 에 리스트(list)처럼 직접 접근(index)할 수 있습니다: 
# training_data[index]. matplotlib 을 사용하여 
# 학습 데이터의 일부를 시각화해보겠습니다.

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

# Dataset 을 DataLoader 의 인자로 전달합니다. 
# 이는 데이터셋을 순회 가능한 객체(iterable)로 감싸고, 
# 자동화된 배치(batch), 샘플링(sampling), 섞기(shuffle) 및 
# 다중 프로세스로 데이터 불러오기(multiprocess data loading)를 지원합니다. 
# 여기서는 배치 크기(batch size)를 64로 정의합니다. 
# 즉, 데이터로더(dataloader) 객체의 각 요소는 
# 64개의 특징(feature)과 정답(label)을 묶음(batch)으로 반환합니다.

# batch_size = 64

# # 데이터로더를 생성합니다.
# train_dataloader = DataLoader(training_data, batch_size=batch_size)
# test_dataloader = DataLoader(test_data, batch_size=batch_size)

# for X, y in test_dataloader:
#     print("Shape of X [N, C, H, W]: ", X.shape)
#     print("Shape of y: ", y.shape, y.dtype)
#     break

# # 모델 만들기
# # PyTorch에서 신경망 모델은 
# # nn.Module 을 상속받는 클래스(class)를 생성하여 정의합니다. 
# # __init__ 함수에서 신경망의 계층(layer)들을 정의하고 
# # forward 함수에서 신경망에 데이터를 어떻게 전달할지 지정합니다. 
# # 가능한 경우 GPU로 신경망을 이동시켜 연산을 가속(accelerate)합니다.

# # 학습에 사용할 CPU나 GPU 장치를 얻습니다.
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using {device} device")

# # 모델을 정의합니다.
# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super(NeuralNetwork, self).__init__()
#         self.flatten = nn.Flatten()
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(28*28, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, 10)
#         )

#     def forward(self, x):
#         x = self.flatten(x)
#         logits = self.linear_relu_stack(x)
#         return logits

# model = NeuralNetwork().to(device)
# print(model)

# # 모델 매개변수 최적화하기
# # 모델을 학습하려면 손실 함수(loss function) 와 
# # 옵티마이저(optimizer) 가 필요합니다.

# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

