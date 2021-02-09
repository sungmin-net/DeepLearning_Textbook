import numpy as np

# 기울기 a 와 y 절편 b
fake_a_b = [3, 76]

# x 와 y 의 데이터 값
data = [[2, 81], [4, 93], [6, 91], [8, 97]]
x = [i[0] for i in data]
y = [i[1] for i in data]

def predict(x):
    return fake_a_b[0] * x + fake_a_b[1]

# MSE 함수(y_hat 의 y 의 실제 값, y 는 예측값. 오차 = 예측값 - 실제값)
# 이 함수에 배열 두 개를 넣고, 다 더하고 나누는 식 없이 평균 함수 mean()를 바로 부르면 동작함.-_-?
def mse(y, y_hat):  
    return ((y - y_hat) ** 2).mean()

# MSE 함수를 각 y 값에 대입하여 최종 값을 구하는 함수
def mse_val(y, predict_result):
    return mse(np.array(y), np.array(predict_result))

# 예측 값이 들어갈 빈 리스트
predict_result = []

# 모든 x 값을 한번씩 대입하여, predict_result 의 리스트를 완성
for i in range(len(x)):
    predict_result.append(predict(x[i]))
    print("공부한 시간 = %.f, 실제 점수 = %.f, 예측 점수 = %.f" % (x[i], y[i], predict(x[i])))
    
# 최종 MSE 출력
print("mse 최종 값 : " + str(mse_val(predict_result, y)))