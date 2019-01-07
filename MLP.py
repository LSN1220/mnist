import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.callbacks import ModelCheckpoint

# load mnist dataset
f = np.load('mnist.npz')
x_train, y_train = f['x_train'], f['y_train']
x_test, y_test = f['x_test'], f['y_test']
f.close()
print('训练数据集样本数： %d ,标签个数 %d ' % (len(x_train), len(y_train)))
print('测试数据集样本数： %d ,标签个数  %d ' % (len(x_test), len(y_test)))
print(x_train.shape)
print(y_train[0])

# Scale the pixel to 0-1
x_train = x_train.astype('float')/255
x_test = x_test.astype('float')/255

y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

# Build model
model = Sequential()
model.add(Flatten(input_shape=x_train.shape[1:]))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

# Print model
model.summary()

# Compilation model
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Train model & save the best weights
checkpointer = ModelCheckpoint(filepath='mnist.model.best.hdf5',
                               verbose=1,
                               save_best_only=True)
hist = model.fit(x_train, y_train, batch_size=128, epochs=10,
                 validation_split=0.2, callbacks=[checkpointer],
                 verbose=1, shuffle=True)

# Test model
model.load_weights('mnist.model.best.hdf5')

score = model.evaluate(x_test, y_test, verbose=0)
accuracy = 100 * score[1]
print('Test accuracy: %.4f%%' % accuracy)
