# import json
# import numpy as np
# import keras
# from keras.models import Sequential
# from keras.layers import Conv2D, MaxPool2D, Dense, Flatten
#
# sec = {'-': 7, 'G': 0, 'H': 1, 'I': 2, 'T': 3, 'E': 4, 'B': 5, 'S': 6}
#
# input = []
# output = []
#
# with open('temp.json', 'r') as f:
#     datas = json.load(f)
#
#     np.random.shuffle(np.arange(datas.__len__()))
#
#     for data in datas[:200]:
#         input.append(data['input'])
#         output.append(sec[data['output']])
#
# input = np.reshape(input, (-1, 1, 40, 25))
# output = keras.utils.to_categorical(output, num_classes=8)
#
# x_train = input[:180]
# y_train = output[:180]
# x_test = input[180:]
# y_test = output[180:]
#
# plato = Sequential()
# plato.add(Conv2D(720, kernel_size=11, strides=1, padding='valid',
#                  input_shape=(1, 40, 25), data_format="channels_first"))
# plato.add(MaxPool2D(pool_size=4, strides=1))
# plato.add(Conv2D(720, kernel_size=2, strides=1, padding='valid'))
# plato.add(MaxPool2D(pool_size=2, strides=1))
# plato.add(Conv2D(512, kernel_size=2, strides=1))
# plato.add(MaxPool2D(pool_size=2, strides=1))
# plato.add(Conv2D(256, kernel_size=2, strides=1))
# plato.add(MaxPool2D(pool_size=2, strides=1))
# plato.add(Conv2D(128, kernel_size=2, strides=1))
# plato.add(MaxPool2D(pool_size=2, strides=1))
# plato.add(Conv2D(48, kernel_size=2, strides=1))
# plato.add(Flatten())
# plato.add(Dense(256))
# plato.add(Dense(8, activation='softmax'))
# plato.summary()
#
# plato.compile(loss='categorical_crossentropy',
#               optimizer='sgd',
#               metrics=['accuracy'])
# plato.fit(x_train, y_train, batch_size=1, epochs=3,
#           verbose=1, validation_data=[x_test, y_test])
#
# val_loss, val_acc = plato.evaluate(x_test, y_test)
# print(val_loss, val_acc)
#

import json
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Embedding
import matplotlib.pyplot as plt

model = Sequential()
model.add(Embedding(1000, 10, input_length=10))
# 模型将输入一个大小为 (batch, input_length) 的整数矩阵。
# 输入中最大的整数（即词索引）不应该大于 999 （词汇表大小）
# 现在 model.output_shape == (None, 10, 64)，其中 None 是 batch 的维度。

# input_array = np.random.randint(1000, size=(32, 10))
input_array = np.eye(10)

model.compile('rmsprop', 'mse')
output_array = model.predict(input_array)
# assert output_array.shape == (32, 10, 64)


print(input_array)
print('=============')
print(output_array.shape)

plt.imshow(output_array[8])
plt.show()

# plt.imshow(input_array)
# plt.show()

