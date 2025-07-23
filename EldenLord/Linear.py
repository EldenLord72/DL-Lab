#Simple Reg
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense


X = np.linspace(0,10,1000)
Y = 3*X+7+(3*np.random.randn(1000))

model = Sequential()
model.add(Dense(units=1,input_dim=1,activation='linear'))

model.compile(optimizer='sgd',loss='mse',metrics=['mae'])
model.fit(X,Y,epochs=20)
model.evaluate(X,Y)
result=model.predict(X)

plt.scatter(X,Y,label='Origin data',color='black')
plt.plot(X,result,label='Predicted data',color='red')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Lin Reg')
plt.show()

print("Tarnished")
