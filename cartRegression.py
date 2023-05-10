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

    return dfx, dfy

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
   
    dfx, dfy = readCSV('cLabel.csv')

    trainx, testx, trainy, testy = train_test_split(dfx, dfy, test_size = 0.30)

    cart = generateModel(trainx, trainy)

    score = cart.score(testx, testy)

    print(score)

    # save model

    obj = sio.dumps(cart)

    with open('modelRCart.pkl', 'wb') as f:
        f.write(obj)

    return 0

if __name__ == '__main__':
    main()
