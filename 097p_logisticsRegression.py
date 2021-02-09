import numpy as np
import matplotlib.pyplot as plt

# 공부 시간 x 와 합격 여부 y 의 리스트 만들기
data = [[2,0], [4,0], [6,0], [8,1], [10,1], [12,1], [14,1]]

x_data = [i[0] for i in data]
y_data = [i[1] for i in data]

# 그래프로 나타내기
plt.scatter(x_data, y_data)
plt.xlim(0, 15)
plt.ylim(-0.1, 1.1)


# 기울기 a와 절편 b의 값 초기화
a = 0
b = 0

# 학습률
lr = 0.05

# 시그모이드 함수 정의
def sigmoid(x):
    return 1 / (1 + np.e ** (-x))

# 경사 하강법 실행. 1,000번 반복할 때마다 각 x_data 값에 현재의 a값, b값 출력
for i in range(2001):
    for x_data, y_data in data:
        a_diff = x_data * (sigmoid(a * x_data + b) - y_data)
        b_diff = sigmoid(a * x_data + b) - y_data
        a = a - lr * a_diff
        b = b - lr * b_diff
        if i % 1000 == 0:
            print("epoch = %.f, 기울기 = %.04f, 절편 = %.04f" % (i, a, b))

# 앞서 구한 기울기와 절편을 이용해 그래프 그리기
plt.scatter(x_data, y_data)
#plt.xlim(0, 15) # 중복코드
#plt.ylim(-0.1, 1.1) # 중복코드
x_range = (np.arange(0, 15, 0.1)) # 그래프로 나타낼 x 값의 범위 정하기
plt.plot(np.arange(0, 15, 0.1), np.array([sigmoid(a * x  + b) for x in x_range]))
plt.show()
