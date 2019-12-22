# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 18:46:07 2019

@author: Youssef AAMER
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 22:27:41 2019

@author: Youssef AAMER
"""
import keras as keras
import tensorflow as tf
print(tf.__version__)
import pandas as pd
#import xlrd
import numpy as np
from keras.initializers import normal

path    = 'D:\\Maching Learning\\Deep learning\OTT_Radio_Config_DL\\Data_Set\\Dataset_OK_last.xlsx'
Dataset = pd.read_excel(path)


predictions=np.zeros((565,4))
Test_Vector=np.zeros((565,4))
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
#from sklearn.model_selection import cross_validate

#----------------Partie L800------------------------
#---------------------------------------------------

initializer = normal(mean=0, stddev=0.02, seed=30)

x_train, x_test, y_train, y_test = train_test_split(Dataset.iloc[:,0:17], Dataset.iloc[:,17:20], test_size=0.2, random_state=42)
input_dim1 = x_train.shape[1]
 
model = keras.models.Sequential()
model.add(keras.layers.Dense(40,input_dim=input_dim1, activation='relu',kernel_initializer=initializer))
model.add(keras.layers.Dense(40, activation='relu',kernel_initializer=initializer))
model.add(keras.layers.Dense(20, activation='relu',kernel_initializer=initializer))
model.add(keras.layers.Dense(3, activation='sigmoid',kernel_initializer=initializer))


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)
model.fit(x_train, y_train,batch_size=100,verbose=1,epochs=1000)

#,callbacks=[early_stop]

val_loss_Bands, val_acc_Bands = model.evaluate(x_test, y_test)
print(val_loss_Bands)
print(val_acc_Bands)
model.save('L800_1800_2600.model')
new_model = keras.models.load_model('L800_1800_2600.model')

predictions[:,0:3] = new_model.predict(x_test).reshape(565,3)
Test_Vector[:,0:3] = y_test


#from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler

#----------------Partie PRB Usage------------------------
#---------------------------------------------------

x_train, x_test, y_train, y_test = train_test_split(Dataset.iloc[:,0:17], Dataset.iloc[:,20], test_size=0.2, random_state=42)
input_dim1 = x_train.shape[1]
scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

model_PRB = keras.models.Sequential()
model_PRB.add(keras.layers.Dense(60,input_dim=input_dim1, activation='relu'))
model_PRB.add(keras.layers.Dense(60, activation='relu'))
model_PRB.add(keras.layers.Dense(60, activation='relu'))
model_PRB.add(keras.layers.Dense(1,activation='linear'))

model_PRB.compile(optimizer='adam',
              loss='mae',
              metrics=['mse','mae'])

model_PRB.fit(x_train, y_train,validation_data=(x_test, y_test),epochs=1500)
val_loss_PRB , val_acc_PRB,val_acc_PRB = model_PRB.evaluate(x_test, y_test)

model_PRB.save('PRB_Usage.model')
new_model = keras.models.load_model('PRB_Usage.model')

predictions[:,3] = new_model.predict(x_test).reshape(565,)
Test_Vector[:,3] = y_test.reshape(565,)
np.savetxt("Resultas.csv", predictions, delimiter=",")
np.savetxt("Y_Test_Vector.csv", Test_Vector, delimiter=",")

print('Loss band:',val_loss_Bands)
print('Acc band:', val_acc_Bands)
print('Loss PRB:',val_loss_PRB)
print('Acc PRB:',val_acc_PRB)