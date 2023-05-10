import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import export_graphviz
import skops.io as sio

def readCSV(path):

    df = pd.read_csv(path)
    df.drop('number',  axis=1, inplace=True)
    df.drop('gravidade',  axis=1, inplace=True)

    # split into x and y
    dfy = df['rotulo']
    dfx = df.drop('rotulo', axis = 1)

    trainx, testx, trainy, testy = train_test_split(dfx, dfy, test_size = 0.30)

    # save csvs

    print('Saving csvs...')
    pd.DataFrame(trainx).to_csv('./data/classification/trainx.csv', index=False)
    pd.DataFrame(testx).to_csv('./data/classification/testx.csv', index=False)
    pd.DataFrame(trainy).to_csv('./data/classification/trainy.csv', index=False)
    pd.DataFrame(testy).to_csv('./data/classification/testy.csv', index=False)

    return trainx, testx, trainy, testy

def readFromdData():

    print('Reading from data...')
    testx = pd.read_csv('./data/classification/testx.csv')
    testy = pd.read_csv('./data/classification/testy.csv')
    testx = testx.iloc[:]
    testy = np.ravel(testy)
    trainx = pd.read_csv('./data/classification/trainx.csv')
    trainy = pd.read_csv('./data/classification/trainy.csv')
    trainx = trainx.iloc[:]
    trainy = np.ravel(trainy)

    return trainx, testx, trainy, testy

def generateModel(dfx, dfy):

    rf = RandomForestClassifier(random_state=0)

    params = {
        'max_depth': [5, 6, 7],
        'n_estimators': [50, 100, 200],
        'max_features': ['sqrt', 'log2'],
        'criterion': ['gini', 'entropy']
    }

    rf = GridSearchCV(rf, params, cv = 10)

    rf.fit(dfx, dfy)

    return rf

def main():
   
    if len(sys.argv) < 3:
        print('Usage: python3 cartRegression.py <new|read> <path>')
        return 1
    
    args = sys.argv[1:]

    if args[0] == 'new':
        testx, trainx, testy, trainy = readCSV('cLabel.csv')
    elif args[0] == 'read':
        trainx, testx, trainy, testy = readFromdData()

    rf = generateModel(trainx, trainy)

    score = rf.score(testx, testy)

    print(score)

    count = 0

    for estimator in rf.best_estimator_.estimators_:
        path = './plot/randomForest/tree' + str(count) + '.dot'
        export_graphviz(estimator, out_file=path, 
                    feature_names = trainx.columns,
                    rounded = True, proportion = False, 
                    precision = 2, filled = True)
        count += 1

    obj = sio.dumps(rf)


    path = "./models/" + args[1] + '.pkl'

    with open(path, 'wb') as f:
        f.write(obj)

    return 0

if __name__ == '__main__':
    main()
