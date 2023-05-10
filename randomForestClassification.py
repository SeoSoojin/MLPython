from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
import skops.io as sio

def readCSV(path):

    df = pd.read_csv(path)
    df.drop('number',  axis=1, inplace=True)
    df.drop('gravidade',  axis=1, inplace=True)

    # split into x and y
    dfy = df['rotulo']
    dfx = df.drop('rotulo', axis = 1)

    return dfx, dfy

def generateModel(dfx, dfy):

    trainx, testx, trainy, testy = train_test_split(dfx, dfy, test_size = 0.30)

    rf = RandomForestClassifier(random_state=0)

    params = {
        'max_depth': [5, 6, 7, 8, 9, 10],
        'n_estimators': [50, 100, 200],
    }

    rf = GridSearchCV(rf, params, cv = 10)

    rf.fit(trainx, trainy)

def main():
   
    dfx, dfy = readCSV('data.csv')

    rf = generateModel(dfx, dfy)

    # save model

    obj = sio.dumps(rf)

    with open('model.pkl', 'wb') as f:
        f.write(obj)

    return 0

if __name__ == '__main__':
    main()
