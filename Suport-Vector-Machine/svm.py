import numpy as np
from sklearn import preprocessing, cross_validation, neighbors, svm
import pandas as pd




if __name__ == "__main__":
        
    df = pd.read_csv('data/breast-cancer-wisconsin.data')

    df.replace('?', -99999, inplace=True)
    df.drop(['id'], 1, inplace=True) # remove ID

    x = np.array(df.drop(['class'], 1))
    y = np.array(df['class'])

    x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2) # teste de treino
    clf = svm.SVC() # instancia o classificador
    clf.fit(x_train, y_train) # treinar o classificador

    accuracy = clf.score(x_test, y_test) #testar acurácia do classificador, dado o dataset
    print(accuracy)

    # test
    example_measures = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1], [4, 2, 1, 2, 2, 2, 3, 2, 1]])
    example_measures = example_measures.reshape(len(example_measures), -1)

    prediction = clf.predict(example_measures)
    print(prediction) 
    # 98% de certeza se o cancer vai ser maligno ou benigno