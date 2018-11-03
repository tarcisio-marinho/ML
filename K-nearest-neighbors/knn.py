import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd


# muitas libs reconhecem -99999 como um valor fora da linha dos dados

if __name__ == "__main__":
        
    df = pd.read_csv('data/breast-cancer-wisconsin.data')

    df.replace('?', -99999, inplace=True) # limpa os dados inválidos
    df.drop(['id'], 1, inplace=True) # remove ID

    x = np.array(df.drop(['class'], 1)) # features
    y = np.array(df['class']) # classificação das features "labels"

    # suffle the data, separa em 20% dos dados para serem testados 
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2)

    # classificador
    clf = neighbors.KNeighborsClassifier()

    # treinar o classificador
    clf.fit(x_train, y_train) 

    accuracy = clf.score(x_test, y_test)
    print('accuracy: {}'.format(accuracy))

    # maligo, ex: 8,10,10,8,7,10,9,7,1
    # test
    example_measures = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1]])

    # reshape(quantidade_de_predicoes, -1=uma dimensão)
    example_measures = example_measures.reshape(len(example_measures), -1)

    tipos_de_cancer = {4 : "maligo", 2 : "benigno"}

    prediction = clf.predict(example_measures)
    print('Tipo de cancer é: {}'.format(tipos_de_cancer[prediction.item(0)]))

