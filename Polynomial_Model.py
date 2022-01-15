# encoding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import seaborn as sns
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

def polynomialFonk():
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

    #Polynomal featuresla veri model düzenlemesi ile beraber model eğitimi.
    from sklearn.preprocessing import PolynomialFeatures
    poly_reg = PolynomialFeatures(degree=3)
    x_poly = poly_reg.fit_transform(x_train)
    lin_reg = LinearRegression()
    lin_reg.fit(x_poly, y_train)
    y_pred = lin_reg.predict(poly_reg.fit_transform(x_test))

    #Normalize işlemi
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    y_test = sc.fit_transform(y_test)
    y_pred = sc.fit_transform(y_pred)

    degerler = [np.sqrt(mean_squared_error(y_test, y_pred)).mean(),
                mean_absolute_error(y_test, y_pred).mean(),
                r2_score(y_test, y_pred).mean()]

    #Model değerlendirme metrikleri
    print("RMSE :", np.sqrt(mean_squared_error(y_test, y_pred)).mean())
    print("MAE :", mean_absolute_error(y_test, y_pred).mean())
    print("R2 :", r2_score(y_test, y_pred).mean())

    y_test = sc.inverse_transform(y_test)
    y_pred = sc.inverse_transform(y_pred)

    return degerler






