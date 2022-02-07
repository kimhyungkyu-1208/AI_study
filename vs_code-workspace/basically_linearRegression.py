# 선형회귀는 입력(x_train)과 정답(y_train) 데이터를 학습하여 
# 기울기와 편향(W,b) 값을 구하고, 
# 미지의 입력(x)이 들어왔을 때 어떤 출력(y)이 나올지 예상하는 지도학습 머신러닝

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1)    #random value initialize

x_train = torch.FloatTensor([[1], [2], [3]])    # input
y_train = torch.FloatTensor([[3], [6], [9]])    # answer data

# 파이토치는 텐서를 선언하고 자동미분을 통해 학습하는 구조입니다. 
# 선형회귀 기본 가설인 y = Wx + b 에서 우리가 구하고자 하는 W와 b를 텐서로 초기화 해줍니다. 
# requires_grad=True 를 갖는 2개의 텐서(tensor) W 와 b 를 만듭니다.

# requires_grad=True 는 모든 연산(operation)들을 추적해야 한다고 알려주는 parameter 입니다.
W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
hypothesis = x_train * W + b
# 손실함수는 평균제곱오차, 옵티마이저는 SGD(Stochastic Gradient Descent)를 사용

# 손실 함수는 실제값과 예측값의 차이(loss, cost)를 수치화해주는 함수입니다. 
# 오차가 클수록 손실 함수의 값이 크고, 오차가 작을수록 손실 함수의 값이 작아집니다. 
# 선형회귀란 손실 함수의 값을 최소화 하는 W, b를 찾아가는것이 학습 목표이다. 
# 일반적으로 회귀문제에서는 평균제곱오차
# 분류 문제에서는 크로스 엔트로피를 사용합니다.

# 손실함수를 줄여나가면서 학습하는 방법은 어떤 optimizer를 사용하느냐에 따라 달라집니다. 
# 옵티마이저는 학습 데이터(Train data)셋을 이용하여 모델을 학습 할 때 
# 데이터의 실제 결과와 모델이 예측한 결과를 기반으로 잘 줄일 수 있게 만들어주는 역할을 하는 것입니다. 
# 딥러닝에서 모델을 학습시킨다는건 최적화(optimization) 태스크를 수행하는 것과 같습니다. 
# 여기서 최적화란, 손실 함수(loss funciton)의 최솟값을 찾아나가는 일련의 과정을 말합니다. 
# 최적화는 각 학습 단계에서 모델의 오류를 줄이기 위해 모델 매개변수를 조정하는 과정입니다. 
# 최적화 알고리즘은 이 과정이 수행되는 방식
# (여기에서는 확률적 경사하강법(SGD; Stochastic Gradient Descent))을 정의합니다. 
# 한 스텝마다 이동하는 크기, 즉 보폭이 학습률(learning rate)로 정의되고, 
# 앞으로 이동할 방향은 현 지점의 기울기(gradient)를 통해 정의됩니다.

# ’SGD’는 경사 하강법의 일종입니다. 
# lr은 학습률(learning rate)를 의미합니다. 
# Stochastic Gradient Desenct(SGD)는 Loss Function을 계산할 때, 
# 전체 데이터(Batch) 대신 일부 데이터의 모음(Mini-Batch)를 사용하여 
# Loss Function을 계산하여 속도가 빠르게 동작하는 옵티마이저 입니다.

# 학습하려는 모델의 매개변수와 학습률(learning rate) 하이퍼파라매터를 등록하여 옵티마이저를 초기화
cost = torch.mean((hypothesis - y_train) ** 2)
optimizer = optim.SGD([W, b], lr=0.01)

# 최적화 단계의 각 반복(iteration)을 에폭이라고 부릅니다. 하나의 에폭은 다음 두 부분으로 구성됩니다.

# 1. 학습 단계(train loop) 
#   - 학습용 데이터셋을 반복(iterate)하고 최적의 매개변수로 수렴합니다.
# 2. 검증/테스트 단계(validation/test loop) 
#   - 모델 성능이 개선되고 있는지를 확인하기 위해 테스트 데이터셋을 반복(iterate)합니다.

# epoch 를 100으로 반복 학습 합니다. 
# 모델의 예측값과 그에 해당하는 정답(label)을 사용하여 오차(error, 손실(loss, cost) )를 계산합니다. 
# Pytorch에서는 gradients값들을 추후에 backward를 해줄때 계속 더해주기 때문"
# 에 우리는 항상 backpropagation을 하기전에 gradients를 zero로 만들어주고 시작을 해야합니다. 
# optimizer.zero_grad()를 호출하여 모델 매개변수의 변화도를 0으로 설정합니다. 
# 기본적으로 변화도는 더해지기(add up) 때문에 중복 계산을 막기 위해 반복할 때마다 명시적으로 0으로 설정합니다. 
# 한번의 학습이 완료되어지면(즉, Iteration이 한번 끝나면) gradients를 항상 0으로 만들어 주어야 합니다. 
# 만약 gradients를 0으로 초기화해주지 않으면 gradient가 의도한 방향이랑 다른 방향을 가르켜 학습이 원하는 방향으로 이루어 지지 않습니다.

# 다음 단계는 신경망을 통해 이 예측 손실(prediction loss)을 역전파합니다. 
# 오차 텐서(error tensor)에 .backward() 를 호출하면 역전파가 시작됩니다. 
# 역전파 계산은 .backward()를 호출하여, 자동으로 모든 기울기(gradient)를 계산할 수 있습니다. 
# PyTorch는 각 매개변수에 대한 손실의 변화도를 자동 저장합니다.

# 마지막으로 .step 을 호출하여 경사하강법(gradient descent)을 시작합니다. 
# 옵티마이저는 .grad 에 저장된 변화도에 따라 각 매개변수를 조정합니다. 
# tensor에 대한 기울기(gradient)는 .grad 속성에 누적될 것입니다.

# 경사하강법으로 학습합니다. 
# 비용함수는 평귭제곱오차를 사용합니다. 
# W = 3, b = 0 에 가까워 짐을 확인 할 수 있습니다.

nb_epochs = 2800 # 원하는만큼 경사 하강법을 반복
for epoch in range(nb_epochs + 1):
    hypothesis = x_train * W + b
    cost = torch.mean((hypothesis - y_train) ** 2)
    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(epoch, nb_epochs, W.item(), b.item(), cost.item()))

# 우리는 이제 학습된 W 와 b 값으로 새로운 입력에 대한 예측을 할 수 있습니다. 
# 기록을 추적하는 것(메모리를 사용하는 것)을 방지하기 위해 
# with torch.no.grad(): 로 코드 block을 감쌀 수 있습니다. 
# 새로운 입력 4에 대해 예측값 y를 계산해서 pred_y를 구할 수 있습니다.

test_var =  torch.FloatTensor([[4.0]]) 
# 입력한 값 4에 대해서 예측값 y를 계산해서 pred_y에 저장
with torch.no_grad():
    pred_y = test_var * W + b
    print("훈련 후 입력이 4일 때의 예측값 :", pred_y) 
