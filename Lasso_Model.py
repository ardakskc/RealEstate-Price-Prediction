# encoding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LassoCV
import seaborn as sns
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore'),

def LassoFonk():
    #Datadan fiyat kolonunu ayırdım.
    data = pd.read_csv("Real_Estate.csv")
    price = data.iloc[:,0]
    satir_sayisi = price.count()
    price = pd.DataFrame(data=price,index=range(satir_sayisi),columns=['price'])
    data.drop('price', axis=1, inplace=True)
    data = pd.DataFrame(data=data,index=range(satir_sayisi), columns=data.columns)
    print(price.head())
    print(data.head())

    #Train-Test ayrımı
    x_train, x_test,y_train,y_test = train_test_split(data,price,test_size=0.33, random_state=0)
    #Normalize işlemi
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(x_train)
    X_test = sc.fit_transform(x_test)
    Y_train = sc.fit_transform(y_train)
    Y_test = sc.fit_transform(y_test)

    #Modeli oluşturdum
    #Amaç hata kareler toplamını minimize eden katsayıları, katsayılara ceza uygularayarak bulmaktır.->Alpha parametresi.
    #Hatayı Azaltmak için hangi lambdadeğeri en az hatayı verir test edilir. LassoCV en iyi parametreli modeli döndürür.
    lambdalar2 = 10**np.linspace(10, -2, 100)*0.5

    lasso_CV = LassoCV(alphas=lambdalar2, cv=10, max_iter=10000).fit(X_train, Y_train)
    print("Alpha:" ,lasso_CV.alpha_)
    #En iyi alpha değerini modelin parametresi olarak set ederim.
    lasso_tuned = Lasso().set_params(alpha = lasso_CV.alpha_).fit(X_train, Y_train)
    y_pred = lasso_tuned.predict(X_test)

    degerler = [np.sqrt(mean_squared_error(Y_test, y_pred)).mean(),
                mean_absolute_error(Y_test, y_pred).mean(),
                r2_score(Y_test, y_pred).mean()]

    #Model değerlendirme metrikleri
    print("RMSE :",np.sqrt(mean_squared_error(Y_test,y_pred)).mean())
    print("MAE :",mean_absolute_error(Y_test,y_pred).mean())
    print("R2 :",r2_score(Y_test,y_pred).mean())

    return degerler


