import numpy as np

# 가중치와 바이어스
w11 = np.array([-2, -2])
w12 = np.array([2, 2])
w2 = np.array([1, 1])
b1 = 3
b2 = -1
b3 = -1

def multiLayerPerceptron(x, w, b):
    y = np.sum(w * x) + b
    if y <= 0:
        return 0
    else :
        return 1

# NAND 게이트
def nandGate(x1, x2):
    return multiLayerPerceptron(np.array([x1, x2]), w11, b1)

# OR 게이트
def orGate(x1, x2):
    return multiLayerPerceptron(np.array([x1, x2]), w12, b2)

# AND 게이트
def andGate(x1, x2):
    return multiLayerPerceptron(np.array([x1, x2]), w2, b3)

# XOR 게이트
def xorGate(x1, x2):
    return andGate(nandGate(x1, x2), orGate(x1, x2))

if __name__ == '__main__':
    for x in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        y = xorGate(x[0], x[1])
        print("입력 값 : " + str(x) + ", 출력 값 : " + str(y))