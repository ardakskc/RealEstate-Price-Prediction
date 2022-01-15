# encoding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

REGION_ID = 2661  # Saint Petersburg ID

MIN_AREA = 20  # Daire metrekaresi için Outlier aralığı
MAX_AREA = 200

MIN_KITCHEN = 6  # Mutfak metrekaresi için Outlier aralığı
MAX_KITCHEN = 30

MIN_PRICE = 1_500_000  # Fiyat için Outlier Aralığı
MAX_PRICE = 50_000_000


def dataset_filter(df: pd.DataFrame) -> pd.DataFrame:
    # Saat kolonunu sildim.
    df.drop('time', axis=1, inplace=True)
    df['date'] = pd.to_datetime(df['date'])  # Veri setindeki zamanın dönüşümü
    # Oda kolonu stüdyo daireler için -1, -2 değeri içeriyordu. Bunları düzenledim.
    df['rooms'] = df['rooms'].apply(lambda x: 0 if x < 0 else x)
    df['price'] = df['price'].abs()  # Negatif değerlerin mutlak değeri.
    # Fiyat ve metrekare değerleri için outlier temizledim.
    df = df[(df['area'] <= MAX_AREA) & (df['area'] >= MIN_AREA)]
    df = df[(df['price'] <= MAX_PRICE) & (df['price'] >= MIN_PRICE)]

    # Mutfak metrekare değerleri için outlier temizledim.
    df.loc[(df['kitchen_area'] >= MAX_KITCHEN) | (df['area'] <= MIN_AREA), 'kitchen_area'] = 0

    # Ev metrekaresi üzerinden mutfak metrekaresinin düzenlenmesi.
    erea_mean, kitchen_mean = df[['area', 'kitchen_area']].quantile(0.5)
    kitchen_share = kitchen_mean / erea_mean
    df.loc[(df['kitchen_area'] == 0) & (df['rooms'] != 0), 'kitchen_area'] = \
        df.loc[(df['kitchen_area'] == 0) & (df['rooms'] != 0), 'area'] * kitchen_share
    df['kitchen_area']=df['kitchen_area'].round(2)#Virgülden sonra 2 basamak alınsın.

    df.loc[df['object_type'] == 11,'object_type'] = 2  # iki farklı obje tipi var idi biri 11 biri 1 ile kodlanmıştı. 1-2 şeklinde değiştirdim.
    return df


def region_filter(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df['region'] == REGION_ID]
    df.drop('region', axis=1, inplace=True)
    print(f'Selected {len(df)} samples in region {REGION_ID}.')
    return df


def features_update(df: pd.DataFrame) -> pd.DataFrame:
    # Date kolonunu sene ve ay olarak böldüm.
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df.drop('date', axis=1, inplace=True)
    # Dairenin katıyla binanın kaç katlı oldugu arasındaki ilişki
    df['level_to_levels'] = (df['level'] / df['levels'])
    df['level_to_levels']=df['level_to_levels'].round(4)#Virgülden sonra 4 basamak alınsın.
    # Dairenin metrekaresi üzerinden ortalama odaya düşen metrekare hesabı.
    df['area_to_rooms'] = (df['area'] / df['rooms']).abs()
    # 0'ile bölümü engellemek.
    df.loc[df['area_to_rooms'] == np.inf, 'area_to_rooms'] = \
        df.loc[df['area_to_rooms'] == np.inf, 'area']
    df['area_to_rooms']=df['area_to_rooms'].round(2)#Virgülden sonra 2 basamak alınsın.
    return df

def basla():
#Veri setini dataframe içine alıp, update işlemleini gerçekleştirdim.
    data = pd.read_csv("all_v2.csv")
    print(f'Data shape: {data.shape}')
    data = data.pipe(dataset_filter)
    print(f'Data shape: {data.shape}')
    data = data.pipe(region_filter)
    data = data.pipe(features_update)
    print(f'Data shape: {data.shape}')
    #Güncellenmiş datayı yeni bir csv dosyası içine kaydettim.
    data.to_csv("Real_Estate.csv", index=False)
    return data

#Senelere göre ortalama fiyat grafiği.
def DataSetGoster(data: pd.DataFrame) -> pd.DataFrame:
    x = []
    y = [2018,2019,2020,2021]
    for i in range(2018,2022):
        temp = data[data['year'] == i]
        price = temp.iloc[:, 0]
        x.append(np.mean(price))

    plt.xlabel('Years')
    plt.ylabel('Million Ruble')
    plt.plot(y, x)
    plt.show()


    #Fiyat kolonuna , featuresların etkisi için korelasyon matrisi
def DataSetGoster2(data: pd.DataFrame) -> pd.DataFrame:
    correlation = data.corr()
    ax = sns.heatmap(correlation, center=0, cmap='RdBu_r')
    l, r = ax.get_ylim()
    ax.set_ylim(l + 0.5, r - 0.5)
    plt.yticks(rotation=0)
    plt.title('Correlation Matrix')
    plt.show()

