import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
import skops.io as sio

def readCSV(path):

    df = pd.read_csv(path)
    df.drop('number',  axis=1, inplace=True)
    df.drop('rotulo',  axis=1, inplace=True)
    df.drop('pSist', axis=1, inplace=True)
    df.drop('pDiast', axis=1, inplace=True)

    # split into x and y
    dfy = df['gravidade']
    dfx = df.drop('gravidade', axis = 1)

    return dfx, dfy

def generateModel(dfx, dfy):

    clf = MLPRegressor(random_state=1, max_iter=2000)

    params = {
        'learning_rate': ['invscaling', 'adaptive'],
        'solver': ['sgd', 'adam'],
        'hidden_layer_sizes': [(100,), (200,), (300,)],
        'alpha': [1e-5, 1e-4, 1e-3],
    }

    clf = GridSearchCV(clf, params, cv = 10)

    clf.fit(dfx, dfy)

    return clf


def main():
   
    dfx, dfy = readCSV('cLabel.csv')

    trainx, testx, trainy, testy = train_test_split(dfx, dfy, test_size = 0.30)

    rf = generateModel(trainx, trainy)

    score = rf.score(testx, testy)

    print(score)

    # save model

    obj = sio.dumps(rf)

    with open('modelR.pkl', 'wb') as f:
        f.write(obj)

    return 0

if __name__ == '__main__':
    main()
