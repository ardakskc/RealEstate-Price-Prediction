
from matplotlib import pyplot
from sklearn.datasets import load_iris, make_regression
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2, RFECV, RFE, f_regression
from sklearn import linear_model
from sklearn.linear_model import LinearRegression

# feature selection
from sklearn.model_selection import train_test_split


def select_features(X_train, y_train, X_test):
    # configure to select all features
    # en iyi korealasyona sahip k adet değişkeni sıralar
    fs = SelectKBest(score_func=f_regression, k='all')
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)
    # transform test input data
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs

def feature():
    df = pd.read_csv("Real_Estate.csv")
    price = df.iloc[:, 0]
    satir_sayisi = price.count()
    price = pd.DataFrame(data=price, index=range(satir_sayisi), columns=['price'])
    data = df.iloc[:,1:]
    data = pd.DataFrame(data=data, index=range(satir_sayisi), columns=data.columns)

    # Train-Test ayrımı
    x_train, x_test, y_train, y_test = train_test_split(data, price, test_size=0.33, random_state=0)

    # Normalize işlemi
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    X_train = sc.fit_transform(x_train)
    X_test = sc.fit_transform(x_test)
    Y_train = sc.fit_transform(y_train)
    Y_test = sc.fit_transform(y_test)

    reg = LinearRegression()
    reg.fit(X_train, Y_train)

    y_pred = reg.predict(X_test)

    # feature selection
    X_train_fs, X_test_fs, fs = select_features(X_train, Y_train, X_test)
    col = list(data.columns.values)
    # featureların skorlarını sıralar.
    for i in range(len(fs.scores_)):
        print('%s: %f' % (col[i], fs.scores_[i]))
        if(fs.scores_[i]<100):
            df.drop(col[i],axis=1,inplace=True)
    # skorları plotlamak.
    pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
    pyplot.show()

    df.to_csv("Real_Estate.csv", index=False)
    return fs.scores_