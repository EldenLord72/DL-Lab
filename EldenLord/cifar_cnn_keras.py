from keras.models import Sequential
from keras.layers import Dense,Flatten, Conv2D, MaxPooling2D
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.optimizers import Adam
import matplotlib.pyplot as plt


(x_train,y_train),(x_test,y_test)=cifar10.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()
model.add(Conv2D(8,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten(input_shape=(32,32,3)))
model.add(Dense(units=128,activation='relu'))
model.add(Dense(units=10,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

result=model.fit(x_train,y_train,epochs=250,batch_size=64,validation_data=(x_test,y_test))

loss, accuracy = model.evaluate(x_test,y_test)
print(f"test_accuracy:{accuracy},test loss:{loss}")

#plt.plot(result.history['accuracy'],label='TRAIN ACC',color='red')
#plt.plot(result.history['val_accuracy'],label='TEST ACCURACY')
#plt.legend()
#plt.title("EPOCH vs ACCURACY on TRAIN and TEST DATA")
#plt.show()


