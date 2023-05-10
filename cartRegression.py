import sys
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import GridSearchCV, train_test_split
import skops.io as sio

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

def generateModel(dfx, dfy):

    cart = tree.DecisionTreeRegressor(max_depth=5, random_state=1)

    params = {
        'criterion': ['squared_error', 'friedman_mse', 'poisson', 'absolute_error'],
        'splitter': ['best', 'random'],
        'max_depth': [5, 10, 15, 20, 25, 30],
        'min_samples_split': [2, 5, 10, 15, 20],
    }

    cart = GridSearchCV(cart, params, cv = 10)

    cart.fit(dfx, dfy)

    return cart


def main():
   
    if len(sys.argv) < 3:
        print('Usage: python3 cartRegression.py <new|read> <path>')
        return 1
    
    args = sys.argv[1:]

    if args[0] == 'new':
        testx, trainx, testy, trainy = readCSV('cLabel.csv')
    elif args[0] == 'read':
        trainx, testx, trainy, testy = readFromdData()

    cart = generateModel(trainx, trainy)

    score = cart.score(testx, testy)

    print(score)

    estimator = cart.best_estimator_
    
    tree.export_graphviz(estimator, out_file='./plot/cart/' + args[1] + '.dot', 
                    feature_names = trainx.columns,
                    rounded = True, proportion = False, 
                    precision = 2, filled = True)
    
    obj = sio.dumps(cart)

    path = "./models/" + args[1] + '.pkl'

    with open(path, 'wb') as f:
        f.write(obj)

    return 0

if __name__ == '__main__':
    main()
