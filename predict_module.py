# modules
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import datetime as dt
import socket
import threading
import numpy as np
import learn_module
import pymysql
from sklearn.cluster import KMeans
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from neo4j.v1 import GraphDatabase

# 1분전 패킷 데이터 찾기
def get_date():
    minute_ago = dt.datetime.now - dt.timedelta(minutes=1)
    return dt.datetime.strftime(minute_ago, "%Y-%m-%d %H:%M:%S")

# 30분전 패킷 데이터 찾기
def get_date2():
    thirty_minute_ago = dt.datetime.now - dt.timedelta(minutes=30)
    return dt.datetime.strftime(minute_ago, "%Y-%m-%d %H:%M:%S")

# Neo4j DB 접근 클래스
class DataAccessObject(object):
    def __init__(self, uri, user, password):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
    # closer
    def close(self):
        self._driver.close()

    # 모든 노드와 관계를 뽑아옵니다.
    def get_data_all(self):
        output = []
        with self._driver.session() as session:
            response = session.write_transaction(self._get_data_all)
        for line in response:
            output.append({'dns': line['n.dns'], 
            'ip': line['n.ip'],
            'destIp': line['r.destIp'],
            'destPort': line['r.destPort'],
            'sourcePort': line['r.sourcePort'],
            'sourceIp': line['r.sourceIp'],
            'protocol': line['r.proto'],
            'length': line['r.length'],
            'count': line['r.count'],
            'timestamp': line['r.timestamp'],
            'payload': line['r.payload']})
        return output

    # 아이디를 특정하여 그 데이터만 뽑아옵니다.
    def get_data_id(self, ids):
        output = []
        with self._driver.session() as session:
            response = session.write_transaction(self._get_data_id, ids)
        for line in response:
            output.append({'dns': line['n.dns'], 
            'ip': line['n.ip'],
            'destIp': line['r.destIp'],
            'destPort': line['r.destPort'],
            'sourcePort': line['r.sourcePort'],
            'sourceIp': line['r.sourceIp'],
            'protocol': line['r.proto'],
            'length': line['r.length'],
            'count': line['r.count'],
            'timestamp': line['r.timestamp'],
            'payload': line['r.payload']})
        return output
    
    # 현재시간 기준 1분전 데이터 가져옵니다.
    def get_minute_date(self, ids):
        output = []
        with self._driver.session() as session:
            timestamp = dt.datetime.strftime(dt.datetime.now(), "%Y-%m-%d %H:%M:%S");
            print(timestamp, get_date())
            response = session.write_transaction(self._get_minute_date, ids, timestamp, get_date())
        for line in response:
            output.append({'dns': line['n.dns'], 
            'ip': line['n.ip'],
            'destIp': line['r.destIp'],
            'destPort': line['r.destPort'],
            'sourcePort': line['r.sourcePort'],
            'sourceIp': line['r.sourceIp'],
            'protocol': line['r.proto'],
            'length': line['r.length'],
            'count': line['r.count'],
            'timestamp': line['r.timestamp'],
            'payload': line['r.payload'],
            'cnt_mean': line['r.cnt_mean'],
            'cnt_max': line['r.cnt_max'],
            'cnt_max_divide_mean': line['r.cnt_max_divide_mean'],
            'minus_Min_Max': line['r.minus_Min_Max'],
            'ip_len_count': line['r.ip_len_count'],
            'std': line['r.std']})
        return output
    
    # 현재시간 기준 30분전 데이터 가져옵니다.
    def get_thirty_minute_date(self, ids):
        output = []
        with self._driver.session() as session:
            timestamp = dt.datetime.strftime(dt.datetime.now(), "%Y-%m-%d %H:%M:%S");
            print(timestamp, get_date2())
            response = session.write_transaction(self._get_thirty_minute_date, ids, timestamp, get_date2())
        for line in response:
            output.append({'dns': line['n.dns'], 
            'ip': line['n.ip'],
            'destIp': line['r.destIp'],
            'destPort': line['r.destPort'],
            'sourcePort': line['r.sourcePort'],
            'sourceIp': line['r.sourceIp'],
            'protocol': line['r.proto'],
            'length': line['r.length'],
            'count': line['r.count'],
            'timestamp': line['r.timestamp'],
            'payload': line['r.payload'],
            'cnt_mean': line['r.cnt_mean'],
            'cnt_max': line['r.cnt_max'],
            'cnt_max_divide_mean': line['r.cnt_max_divide_mean'],
            'minus_Min_Max': line['r.minus_Min_Max'],
            'ip_len_count': line['r.ip_len_count'],
            'std': line['r.std']})
        return output

    #이하 static method
    # 위의 공개된 method의 이름에 _ 하나 붙어있음
    @staticmethod #
    def _get_data_all(tx):
        result = tx.run("MATCH (n)-[r]-() "
                        "RETURN n.dns, n.ip, r.destIp, r.destPort, r.sourcePort, r.sourceIp, r.proto, r.length, r.count, r.timestamp, r.payload")
        return result

    @staticmethod #
    def _get_data_id(tx, ids):
        result = tx.run("MATCH (n:"+ ids +")-[r]-() "
                        "RETURN n.dns, n.ip, r.destIp, r.destPort, r.sourcePort, r.sourceIp, r.proto, r.length, r.count, r.timestamp, r.payload")
        return result
    @staticmethod #
    def _get_minute_date(tx, ids, ts, mg):
        result = tx.run("MATCH (n:"+ ids +")-[r]-() "
                        "WHERE r.timestamp <= $timestamp AND r.timestamp >= $minute_ago "
                        "RETURN n.dns, n.ip, r.destIp, r.destPort, r.sourcePort, r.sourceIp, r.proto, r.length, r.count, r.timestamp, r.payload, r.cnt_mean, r.cnt_max, r.cnt_max_divide_mean, r.minus_Min_Max, r.ip_len_count, r.std", timestamp=ts, minute_ago=mg)
        return result
    @staticmethod #
    def _get_thirty_minute_date(tx, ids, ts, mg):
        result = tx.run("MATCH (n:"+ ids +")-[r]-() "
                        "WHERE r.timestamp <= $timestamp AND r.timestamp >= $thirty_minute_ago "
                        "RETURN n.dns, n.ip, r.destIp, r.destPort, r.sourcePort, r.sourceIp, r.proto, r.length, r.count, r.timestamp, r.payload, r.cnt_mean, r.cnt_max, r.cnt_max_divide_mean, r.minus_Min_Max, r.ip_len_count, r.std", timestamp=ts, thirty_minute_ago=mg)
        return result


# In[ ]:





# In[18]:


# Video/Non-Video 군집화 및 탐지를 수행합니다.
def predict_video(tmp):
    n_data = tmp[['cnt_max', 'cnt_max_divide_mean']]
    scaler = MinMaxScaler(copy=True, feature_range=(0,1))
    scaler.fit(n_data)
    final = scaler.transform(n_data)
    final = pd.DataFrame(final)
    predict = pd.DataFrame(model_video.predict(final))
    predict.columns=['isVIDEO']
      
    result = pd.concat([tmp, predict], axis=1)
        
    return result


# 영상 데이터에 대해 이상 탐지를 수행합니다.
def predict_anomaly(tmp):
    thresolds = [] # 임계값 리스트

    # Length를 기준으로 이상 탐지 수행
    n_data = np.log(tmp['length'])
    lenn = len(n_data)
    
    predict = model_anomaly.predict(start=lenn, end=lenn*2, dynamic = False)
    
    for i in range(len(n_data)):
        min = data_forecast[i] * 0.8
        max = data_forecast[i] * 1.2
        thresolds.append([min, max])
    
    thresolds = pd.DataFrame(thresolds)
    thresolds.columns =['min', 'max']
    tmp = pd.concat([tmp, thresolds], axis=1)
        
    return tmp

# MySQL DB 전송 함수
def send_Data(data):
    db = pymysql.connect(host="117.16.136.73", user='konkuk-user', password='konkuk-user', db='homepolice', charsert='utf8')
    curs = db.cursor()

    # 이상 패킷 전송
    for i in range(len(data)):
        sql = "INSERT INTO homepolice.history (occured_time, ,src_ip, dest_ip, src_port, dest_port, protocol, description, count, account) VALUES (data.iloc[i,9], data.iloc[i, 5], data.iloc[i, 2], data.iloc[i, 4], data.iloc[i, 3], data.iloc[i, 6], 'Abnormal Packet', data.iloc[i, 8], 'test')"
        curs.execute(sql)
        
    # 임계값 전송
    sql = "INSERT INTO homepolice.thresolds (min, max, account, origin, start_time, end_time) VALUES (list(data['min'].values), list(data['max'].values), 'test', data, data.iloc[0, 9], data.iloc[len(data)-1, 9])"
    
    db.commit()
    db.close()


# 전체 이상 탐지 함수
def predict_main(temp):
    result_anomaly = []
    result_normal = []
    
    # 영상 패킷 탐지
    isv = predict_video(temp)
    video_packet = isv[isv.isVIDEO == 1]
  
    if (len(video_packet) / len(isv))*100<= 0.2:
        print('\n=========탐지된 영상 패킷이 없습니다==========\n')
    else:
        # 영상 데이터 중 CCTV 패킷 분리
        isc = video_packet[video_packet.sourceIp == '10.0.0.95' | video_packet.destIp == '10.0.0.95']
        last = predict_anomaly(isc)
        
        # 예측 결과가 이상인 패킷은 result_anomaly 리스트에 추가
    for row in last.itertuples():
        new_data = np.log(row.length)
        if new_data < row.min or new_data > row.max:
            result_anomaly.append(row)
        else:
            result_normal.append(row)
        
        # 이상 탐지된 패킷들을 MySQL DB로 전송
        if (len(result_anomaly) / len(last)) * 100 > 0.2:
            send_Data(result_anomaly)


# 전역 변수
end = False # 쓰레드 종료 변수
model_video = KMeans() # Video 탐지 모델
model_anomaly = [] # 이상 탐지 모델

# 탐지 쓰레드 함수 
def execute_func(neo):
    global end
    if end:
        return 
    
    user = neo.get_minute_date("test")
    user = pd.DataFrame.from_records(user)
    user['timestamp'] = pd.to_datetime(user['timestamp'])
    user['isVIDEO'] = 0
    user['isAnomaly'] = 0
    
    predict_main(user)


# 학습 쓰레드 함수
def learn_func(neo):
    global end
    if end:
        return
    
    user = neo.get_thirty_minute_date("test")
    user = pd.DataFrame.from_records(user)
    user['timestamp'] = pd.to_datetime(user['timestamp'])
    
    model_video = learn_module.learn_video(user)
    user['kmeans1'] = model_video.predict(user['cnt_max', 'cnt_max_divide_mean'])
    user = user[data.kmeans1 == 1]
    model_anomaly = learn_module.learn_anomaly(user)

# 메인 함수
def main():
    neo4j = DataAccessObject("bolt://13.209.93.63:7687", "neo4j", " ") #neo4j 서버와 세션 연결
        
    threading.Timer(1800.0, learn_func, args=[neo4j]).start() # 30분 간격으로 학습 쓰레드 실행
    threading.Timer(20.0, execute_func, args=[neo4j]).start() # 20초 간격으로 탐지 쓰레드 실행


main()