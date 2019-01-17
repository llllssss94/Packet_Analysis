# modules
import pandas as pd
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# 영상/비영상 탐지 K-Means 학습 모델
def learn_video(data):
    KM = KMeans(n_clusters=2, algorithm='auto', random_state=10)
    n_data = data[['cnt_max', 'cnt_max_divide_mean']]
    scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
    scaler.fit(n_data)
    final = scaler.transform(n_data)
    final = pd.DataFrame(final)
    
    KM.fit(final)
    
    return KM


# 이상/정상 패킷 탐지 Seasonal Arima 학습 모델
def learn_anomaly(data):
    result = []
    
    n_data = np.log(data['length'])
    n_data = pd.DataFrame(n_data)
    n_data = n_data.fillna(method='ffill')
    n_data = n_data.fillna(method='bfill')
    
    model = sm.tsa.SARMIAX(n_data, order = (1,0,2), seasonal_order = (0,0,0,1), trend = 'c')
    model_fit = model.fit()
    
    return model_fit