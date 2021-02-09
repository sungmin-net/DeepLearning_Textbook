from numpy import array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding

# 텍스트 리뷰 자료 지정
docs = ['너무재밌네요', '최고예요', '참 잘 만든 영화예요', '추천하고 싶은 영화입니다.',
        '한번 더 보고싶네요', '글쎄요', '별로예요', '생각보다 지루하네요', '연기가 어색해요',
        '재미없어요']

# print(len(docs))

# 긍정 리뷰는 1, 부정 리뷰는 0으로 클래스 지정
classes = array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
# print(len(classes))

# 토큰화
token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
x = token.texts_to_sequences(docs)
print("\n 리뷰 텍스트, 토큰화 결과:\n")

# 패딩. 서로 다른 길이의 데이터를 4로 맞춤
padded_x = pad_sequences(x, 4)
print("\n패딩 결과\n", padded_x) # 이게 코드냐.. 주피터 노트북 보고 치자. TODO > DONE.

# 임베딩에 입력될 단어의 수를 지정
word_size = len(token.word_index) + 1

# 단어 임베딩을 포함하여 딥러닝 모델을 만들고 결과를 출력
model = Sequential()
model.add(Embedding(word_size, 8, input_length = 4))
model.add(Flatten())
model.add(Dense(1, activation = 'sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(padded_x, classes, epochs = 20)
print("\n Accuracy : %.4f" % (model.evaluate(padded_x, classes)[1]))