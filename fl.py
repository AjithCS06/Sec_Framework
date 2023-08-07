
root_path=''
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from keras.utils import np_utils
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten, BatchNormalization
from tensorflow.keras.layers import Convolution2D, Conv1D
from tensorflow.keras.layers import MaxPooling2D, MaxPooling1D
from keras import backend as K
from keras import backend
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import time
import os
import psutil
import csv
from itertools import repeat
from PIL import Image
from numpy import asarray
from combine import *
from p_aes import *

dfDS=combine_data()

X_full = dfDS.iloc[:, 1:len(dfDS.columns)].values
Y_full = dfDS["STATUS"].values

print(dfDS["STATUS"].value_counts())

xTrain, xTest, yTrain, yTest = train_test_split(X_full, Y_full, test_size=0.10, random_state=123) 
algoName='CNN' #CNN, ANN, DNN

label_encoder = LabelEncoder()
yTrain = label_encoder.fit_transform(yTrain)
yTest = label_encoder.transform(yTest)


xTrain = xTrain.astype('float32')
xTest = xTest.astype('float32')
xTrain = xTrain / 255.
xTest = xTest / 255.

if(algoName=='CNN'):
    xTrain = np.expand_dims(xTrain, axis=2)
    xTest = np.expand_dims(xTest, axis=2)

outputClasses=len(set(Y_full))
#One hot encoding
yTrain = np.array(to_categorical(yTrain))
yTest = np.array(to_categorical(yTest))
print("xTrain", xTrain.shape, "yTrain", yTrain.shape)
print("xTest", xTest.shape, "yTest", yTest.shape)

# FOR TEST SPLIT
xServer, xClients, yServer, yClients = train_test_split(xTrain, yTrain, test_size=0.80,random_state=523) 

def my_metrics(y_true, y_pred):
    accuracy=accuracy_score(y_true, y_pred)
    precision=precision_score(y_true, y_pred,average='weighted')
    recall=recall_score(y_true, y_pred,average='weighted')
    f1Score=f1_score(y_true, y_pred, average='weighted') 
    print("Accuracy  : {}".format(accuracy))
    print("Precision : {}".format(precision))
    print("Recall : {}".format(recall))
    print("f1Score : {}".format(f1Score))
    cm=confusion_matrix(y_true, y_pred)
    print(cm)
    return accuracy, precision, recall, f1Score

verbose, epochs, batch_size = 0, 20, 64
activationFun='relu'
optimizerName='Adam'
def createDeepModel():
    model = Sequential()
    
    if(algoName=='CNN'):
        model.add(Conv1D(filters=5, kernel_size=3, activation=activationFun, input_shape=(13, 1)))
        model.add(Dropout(0.1))
        model.add(MaxPooling1D(pool_size=3))
        model.add(BatchNormalization())
        
        
        model.add(Flatten())
        model.add(Dense(100, activation=activationFun))
        model.add(Dense(50, activation=activationFun))
        model.add(Dense(30, activation=activationFun))
        model.add(Dense(outputClasses, activation='softmax'))
        model.compile(loss='mean_squared_error', optimizer=optimizerName, metrics=['accuracy'])
        
    return model

def predictTestData(yPredict, yTest):
    #Converting predictions to label
    print("yPredict",len(yPredict))
    pred = list()
    for i in range(len(yPredict)):
        pred.append(np.argmax(yPredict[i]))
    #Converting one hot encoded test label to label
    test = list()
    for i in range(len(yTest)):
        test.append(np.argmax(yTest[i]))
    return my_metrics(test, pred)

def sumOfWeights(weights):
    return sum(map(sum, weights))

def getWeights(model):
    allLayersWeights=deepModel.get_weights()
    return allLayersWeights
    
# Initially train central deep model
deepModel=createDeepModel()
numOfIterations=3
numOfClients=5 # 10, 15, 20, 25, 30, 35, 40, 45, 50
modelLocation=root_path+"Models/"+str(algoName)+"_Sync_users_"+str(numOfClients)+"_"+activationFun+"_"+optimizerName+"_FL_Model.h5"
accList, precList, recallList, f1List = [], [], [], []

deepModelAggWeights=[]
firstClientFlag=True

def updateServerModel(clientModel, clientModelWeight):
    global firstClientFlag
    for ind in range(len(clientModelWeight)):
        if(firstClientFlag==True):
            deepModelAggWeights.append(clientModelWeight[ind])            
        else:
            deepModelAggWeights[ind]=(deepModelAggWeights[ind]+clientModelWeight[ind])

def updateClientsModels():
    global clientsModelList
    global deepModel
    clientsModelList.clear()
    for clientID in range(numOfClients):
        m = keras.models.clone_model(deepModel)
        m.set_weights(deepModel.get_weights())
        clientsModelList.append(m)
    
# ----- 1. Train central model initially -----
def trainInServer():
    deepModel.fit(xServer, yServer, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # deepModel.fit(X_full, Y_full, epochs=epochs, batch_size=batch_size, verbose=verbose)
    deepModel.save(modelLocation)
trainInServer()
# ------- 2. Separate clients data into lists ----------
xClientsList=[]
yClientsList=[]
clientsModelList=[]
clientDataInterval=len(xClients)//numOfClients
lastLowerBound=0

for clientID in range(numOfClients):
    xClientsList.append(xClients[lastLowerBound : lastLowerBound+clientDataInterval])
    yClientsList.append(yClients[lastLowerBound : lastLowerBound+clientDataInterval])
    model=load_model(modelLocation)
    clientsModelList.append(model)
    lastLowerBound+=clientDataInterval
# ------- 3. Update clients' model with intial server's deep-model ----------
for clientID in range(numOfClients):
    clientsModelList[clientID].fit(xClientsList[clientID], yClientsList[clientID], epochs=epochs, batch_size=batch_size, verbose=verbose)
        
start_time = time.time()
process = psutil.Process(os.getpid())
for iterationNo in range(1,numOfIterations+1):
    print("Iteration",iterationNo)
    for clientID in range(numOfClients):
        if clientID==0:
            client="user_1"
        elif clientID==1:
            client="user_2"
        elif clientID==2:
            client="user_3"
        elif clientID==3:
            client="user_4"
        elif clientID==4:
            client="user_5"
        print("clientID",clientID)
        print("client name :",client)
        clientsModelList[clientID].compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        clientsModelList[clientID].fit(xClientsList[clientID], yClientsList[clientID], epochs=epochs, batch_size=batch_size, verbose=verbose)
        clientWeight=clientsModelList[clientID].get_weights()
        path='enc_app/static/params/'
        os.makedirs(path, exist_ok=True)
        sname=path+client+".h5"
        print("name-->",sname)
        clientsModelList[clientID].save_weights(sname)
        # Find sum of all client's model
        updateServerModel(clientsModelList[clientID], clientWeight)
        firstClientFlag=False
    #Avarage all clients model
    for ind in range(len(deepModelAggWeights)):
        deepModelAggWeights[ind]/=numOfClients

    dw_last=deepModel.get_weights()

    for ind in range(len(deepModelAggWeights)): 
        dw_last[ind]=deepModelAggWeights[ind]
     
    #Update server's model
    deepModel.set_weights(dw_last) 
    print("Server's model updated")
    print("Saving model . . .")
    deepModel.save(modelLocation)
    # Servers model is updated, now it can be used again by the clients
    updateClientsModels()
    firstClientFlag=True
    deepModelAggWeights.clear()

    yPredict = deepModel.predict(xTest)
    acc, prec, recall, f1Score= predictTestData(yPredict, yTest)
    accList.append(acc)
    precList.append(prec)
    recallList.append(recall)
    f1List.append(f1Score)
    print("Acc:\n", acc)
    print("Prec:\n", prec)
    print("Recall:\n", recall)
    print("F1-Score:\n", f1Score)

memoryTraining=process.memory_percent()
timeTraining=time.time() - start_time
print("---Memory---",memoryTraining)
print("--- %s seconds (TRAINING)---" % (timeTraining))

early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')

history = deepModel.fit(xServer, yServer, epochs=epochs, 
                        validation_data = (xTest,yTest))
                        # callbacks=[early_stopping])

learningAccs=history.history['val_accuracy']
learningLoss=history.history['val_loss']

# resultSaveLocation=root_path+'Results/'+algoName+'_Users_vs_TR_vs_Iterations_vs_AccLossMemTime'+'.csv'
dfSave=pd.DataFrame(columns=['Clients', 'Iterations to converge', 'Accuracy', 'Loss', 'Memory', 'Time'])
dfSaveIndex=0
saveList = [numOfClients, len(learningLoss), learningAccs[len(learningAccs)-1], learningLoss[len(learningLoss)-1], memoryTraining, timeTraining]
dfSave.loc[dfSaveIndex] = saveList

yPredict = deepModel.predict(xTest)
acc, prec, recall, f1Score= predictTestData(yPredict, yTest)

print("Number of users:", numOfClients)
deepModel.save(modelLocation)
print("Epochs:", epochs)
print("BatchSize:", batch_size)
print("Activation:", activationFun, "Optimizer:", optimizerName)

print("Iterations:", numOfIterations)
print("Memory:", memoryTraining)
print("Time:", timeTraining)
print(dfSave)

df_performance_timeRounds = pd.DataFrame(
    {'Accuracy': accList,
     'Precision': precList,
     'Recall': recallList,
     'F1-Score': f1List 
    })
print(df_performance_timeRounds)
perform_encryption()