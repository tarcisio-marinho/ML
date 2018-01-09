import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd

def handle_non_numerical_data(df):
    columns = df.columns.values
    for column in columns:
        text_digit_vals = {} # 'Female' : 0, 'Male' : 1

        def convert_to_int(val):
            return text_digit_vals[val]
        if (df[column].dtype != np.int64 and df[column].dtype != np.float64):
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1
            df[column] = list(map(convert_to_int, df[column]))
    
    return df

if __name__ == "__main__":
    # try to predict/classify without knowing death and survived
    path = "data/titanic.xls" # titanic statistics 

    df = pd.read_excel(path)
    df.drop(['body', 'name'], 1, inplace=True)
    df.convert_objects(convert_numeric=True)
    df.fillna(0, inplace=True)
    print(df.head())

    df = handle_non_numerical_data(df) # convert non numerical data(strings) to integers 
    #print(df.head())
    
    x = np.array(df.drop(['survived'], 1).astype(float))
    y = np.array(df['survived'])
    clf = KMeans(n_clusters=2)
    clf.fit(x)

    correct = 0
    for i in range(len(x)):
        predict_me = np.array()