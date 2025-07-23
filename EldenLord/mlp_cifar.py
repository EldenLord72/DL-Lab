from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.datasets import cifar10
from keras.utils import to_categorical

#import data
(x_train,y_train), (x_test,y_test)=cifar10.lord_dataset

#Categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)