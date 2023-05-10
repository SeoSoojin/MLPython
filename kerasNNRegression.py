from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.model_selection import  GridSearchCV, train_test_split
from scikeras.wrappers import KerasRegressor
import tensorflow as tf

def readCSV(path):

    df = pd.read_csv(path)
    df.drop('number',  axis=1, inplace=True)
    df.drop('rotulo',  axis=1, inplace=True)

    # split into x and y
    dfy = df['gravidade']
    dfx = df.drop('gravidade', axis = 1)

    return dfx.iloc[:], np.ravel(dfy)

def main():

    dfx, dfy = readCSV('cLabel.csv')

    trainx, testx, trainy, testy = train_test_split(dfx, dfy, test_size = 0.30)

    # dataset = tf.data.Dataset.from_tensor_slices((trainx, trainy))
    # test_set = tf.data.Dataset.from_tensor_slices((testx, testy))

    # model = tf.keras.models.load_model('model.keras')


    # predicts = model.predict(testx)
    # print(predicts)

    model = Sequential()

    model.add(Dense(12, input_shape=(5,), activation='relu'))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(1, activation='linear'))

    clf = KerasRegressor(model=model, optimizer='Adam', loss='mean_squared_logarithmic_error', metrics=['mean_squared_logarithmic_error'])
    
    clf.fit(trainx, trainy, epochs=300, batch_size=5, verbose=1)

    score = clf.score(testx, testy)
    print(score)

    # predict 
    predicts = clf.predict(testx)
    print(predicts)

    model.save('modelRegression.h5')

    # _, accuracy = model.evaluate(testx)
    # print('Accuracy: %.2f' % (accuracy*100))

    

if __name__ == '__main__':
    main()
