from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical
from keras.datasets import cifar10
import matplotlib.pyplot as plt

#load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Preprocess the data
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#print(y_train[0])
#plt.imshow(X_train[0])
#plt.show()

# architecture
model = Sequential()
model.add(Flatten(input_shape=(32, 32, 3)))
model.add(Dense(units=10, activation='softmax'))

#compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#train
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)
print(history.history.items())
print(history.history.keys())

#evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy}, Test loss: {loss}')

#visualize
plt.plot(history.history['accuracy'], label='train accuracy',color='blue')
plt.plot(history.history['val_accuracy'], label='validation accuracy',color='red')
plt.legend()
plt.title('Epoch vs Accuracy on Train and Validation data')
plt.show()

