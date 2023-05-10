import sys
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scikeras.wrappers import KerasRegressor
import matplotlib.pyplot as plt

def readCSV(path):

    df = pd.read_csv(path)
    df.drop('number',  axis=1, inplace=True)
    df.drop('rotulo',  axis=1, inplace=True)

    # split into x and y
    dfy = df['gravidade']
    dfx = df.drop('gravidade', axis = 1)

    trainx, testx, trainy, testy = train_test_split(dfx, dfy, test_size = 0.30)

    # save csvs

    print('Saving csvs...')
    pd.DataFrame(trainx).to_csv('./data/regression/trainx.csv', index=False)
    pd.DataFrame(testx).to_csv('./data/regression/testx.csv', index=False)
    pd.DataFrame(trainy).to_csv('./data/regression/trainy.csv', index=False)
    pd.DataFrame(testy).to_csv('./data/regression/testy.csv', index=False)

    return trainx, testx, trainy, testy

def readFromdData():

    print('Reading from data...')
    testx = pd.read_csv('./data/regression/testx.csv')
    testy = pd.read_csv('./data/regression/testy.csv')
    testx = testx.iloc[:]
    testy = np.ravel(testy)
    trainx = pd.read_csv('./data/regression/trainx.csv')
    trainy = pd.read_csv('./data/regression/trainy.csv')
    trainx = trainx.iloc[:]
    trainy = np.ravel(trainy)

    return trainx, testx, trainy, testy

def main():

    if len(sys.argv) < 3:
        print('Usage: python3 kerasNNRegression.py <new|read> <path>')
        return 1
    
    args = sys.argv[1:]

    if args[0] == 'new':
        testx, trainx, testy, trainy = readCSV('cLabel.csv')
    elif args[0] == 'read':
        trainx, testx, trainy, testy = readFromdData()

    model = Sequential()

    model.add(Dense(25, input_shape=(5,), activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='linear'))

    clf = KerasRegressor(model=model, optimizer='Adam', loss='mean_squared_logarithmic_error', metrics=['mean_squared_logarithmic_error'])
    
    clf.fit(trainx, trainy, epochs=300, batch_size=3, verbose=1)

    score = clf.score(testx, testy)
    print(score)

    # plot loss

    loss = clf.history_['loss']

    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, loss, label='Training loss')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./plot/regressionNN/loss.png')

    path = "./models/" + args[1] + '.h5'
    clf.model.save(path)
    

if __name__ == '__main__':
    main()
