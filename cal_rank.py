import pymysql
import pandas as pd
import numpy as np
import datetime
import threading
from pymysql.constants.CLIENT import MULTI_STATEMENTS


# mysql DB로부터 현재 날짜 기준, 이전 7일 간 이상 패킷 데이터를 가져오는 함수
def get_data(curs):
    sql = "SELECT * FROM homepolice.history WHERE date(occured_time) >= date(subdate(now(), INTERVAL 7 DAY)) and date(occured_time) <= date(now())"
    curs.execute(sql)  

    rows = curs.fetchall()
    
    return rows


# 계산된 Security Score를 기반으로 Rank를 정하여 mysql DB로 전송
def send_data(curs, s_score, t_count, u_count,db):
    sql = "INSERT INTO homepolice.ranks (homepolice.ranks.rank, threat_count, unhandled_count, account) VALUE (%s, %s, %s, %s) ON DUPLICATE KEY UPDATE homepolice.ranks.rank=%s, threat_count=%s, unhandled_count=%s"
    data = ( s_score, t_count, u_count, 'test',  s_score, t_count, u_count)
    
    curs.execute(sql, data)
    db.commit()


# Rank를 위한 Security Score 계산 함수
def get_rank():
    db = pymysql.connect(host="117.16.136.73", user='konkuk-user', password='konkuk-user', db='homepolice', charset='utf8', client_flag=MULTI_STATEMENTS, binary_prefix=True)
    curs = db.cursor(pymysql.cursors.DictCursor)
    
    data = get_data(curs)
    len_data = len(data)
    
    count_handled = 0
    count_abnormal_packet = len_data
    abnormal_day = []
    
    for i in range(len_data):
        if data[i]['handled'] == 1:
            count_handled = count_handled + 1
        abnormal_day.append(data[i]['occured_time'].day)
    
    count_abnormal_days = len(set(abnormal_day))

    security_score = 0 # 보안 점수
    handled_score = 0 # 탐지 푸쉬에 대한 응답 없음 횟수
    packet_num_score = 0 # 지난 일주일간 탐지된 이상 패킷의 수
    packet_days_score = 0 # 지난 일주일간 이상 패킷이 탐지된 날의 수
    
    # Security Score 계산 파트
    if count_handled == 10:
         handled_score = 0
    elif count_handled >= 7 and count_handled <= 9:
        handled_score = 25
    elif count_handled >=4 and count_handled <= 6:
        handled_score = 50
    elif count_handled >=1 and count_handled <= 3:
        handled_score = 75
    else:
        handled_score = 100
        
    if count_abnormal_packet >= 100:
        packet_num_score = 0
    elif count_abnormal_packet >= 71 and count_abnormal_packet <= 99:
        packet_num_score = 25
    elif count_abnormal_packet >=41 and count_abnormal_packet <= 70:
        packet_num_score = 50
    elif count_abnormal_packet >=11 and count_abnormal_packet <= 40:
        packet_num_score = 75
    else:
        packet_num_score = 100
    
    if count_abnormal_days == 7:
        packet_days_score = 0
    elif count_abnormal_days >= 6 and count_abnormal_days <= 5:
        packet_days_score = 25
    elif count_abnormal_days >=3 and count_abnormal_days <= 4:
        packet_days_score = 50
    elif count_abnormal_days >=1 and count_abnormal_days <= 2:
        packet_days_score = 75
    else:
        packet_days_score = 100
     
    security_score = 0.2 * handled_score + 0.4 * packet_num_score + 0.4 * packet_days_score
        
    send_data(curs, security_score, count_abnormal_packet, count_handled,db)
    print(datetime.datetime.now())
    curs.close()
    db.close()
    
    threading.Timer(86400.0, get_rank).start() # 하루 간격으로 보안 점수 업데이트


def main():
    get_rank()


main()