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
warnings.filterwarnings('ignore'),

#Data frame'i fiyat ve metrekare olarak böldüm.

def LinearFonk():
    #Doğrusal regresyon, iki değişken arasındaki ilişkiye dayanarak birinin değerini diğerinden tahmin etmeyi sağlayan bir denklem (model) oluşturmayı içerir.
    data = pd.read_csv("Real_Estate.csv")
    price = data.iloc[:, 0]
    data = data.iloc[:, 7]
    price = pd.DataFrame(data=price,index=range(data.count()),columns=['price'])
    data = pd.DataFrame(data=data,index=range(data.count()),columns=['area'])
    print(price.head())
    print(data.head())

    #Train ve test kümesini ayırdım.
    x_train, x_test,y_train,y_test = train_test_split(data,price,test_size=0.33, random_state=0)

    #Standart scaler ile değerleri normalize ettim.
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(x_train)
    X_test = sc.fit_transform(x_test)
    Y_train = sc.fit_transform(y_train)
    Y_test = sc.fit_transform(y_test)


    reg = LinearRegression()
    reg.fit(X_train, Y_train)

    y_pred = reg.predict(X_test)

    degerler = [np.sqrt(mean_squared_error(Y_test,y_pred)).mean(),
                mean_absolute_error(Y_test,y_pred).mean(),
                r2_score(Y_test,y_pred).mean()]
    #Modelin değerlendirme metrikleri.

    # RMSE : Hata kareler ortalamasının kökü, az olması daha iyi
    print("RMSE :",np.sqrt(mean_squared_error(Y_test,y_pred)).mean())
    # MAE : Hataların mutlak değerinin ortalaması, iki değişken arasındaki farkın ölçüsüdür. Az olması daha iyi
    print("MAE :",mean_absolute_error(Y_test,y_pred).mean())
    # R2 : Yüksek olması iyi (diğerlerine göre daha az önemli) , Verilerin regresyon hattına yakınlığı
    print("R2 :",r2_score(Y_test,y_pred).mean())

    y_pred = sc.inverse_transform(y_pred)

    #Modelin grafiği
    plt.scatter(x_train, y_train, color="red")
    plt.plot(x_test, y_pred, color="green")
    plt.title("Linear Regression")
    plt.xlabel("Area\nRMSE: " + str(degerler[0]) + "\nMAE: " + str(degerler[1]) + "\nR2: " + str(degerler[2]))
    plt.ylabel("Price")
    plt.show()


