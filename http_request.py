import requests
import os
import threading

url = "http://117.16.136.91:4443"

payload = "DevEUI=d02544fffef01b61 \
		   &AppEUI=2020202020400000 \
		   &FPort=80 \
		   &DevAddr=1bd20005 \
		   &PHYPayload=BHEyRFbNy51bYp2vBHzXa \
		   &createdTime=2016-11-02%2000%3A00%3A00 \
		   &contentSize=30 \
		   &gwInfo=gwEUI%3A%200000000000000003%20Lati%3A37.38213%20Long%3A127.11654%20Alti%3A78%20i \
		   Rssi%3A1233.5%20Snr%3A254.1%3B%0AgwEUI%3A%200000000000000003%20Lati%3A37.38654%20 \
		   Long%3A127.11517%20Alti%3A70%20Rssi%3A1212.5%20Snr%3A210.1%3B%0 \
		   AgwEUI%3A%200000000000000003%20Lati%3A37.38743%20 \
		   Long%3A127.11234%20Alti%3A72%20Rssi%3A1135.5%20Snr%3A201.1%3B"
headers = {
    'Content-Type': "application/x-www-form-urlencoded",
    'cache-control': "no-cache",
    'Postman-Token': "1f6bd319-7d72-4f1c-8a65-6dbe88b7eaa2"
    }


def send():
    for i in range(0, n):
        try:
            response = requests.request("POST", url, data=payload, headers=headers)

            print(response.text)
        except:
            pass

print("INPUT : ")
n = int(input())

for i in range(0, 5):
	pid = os.fork()

for i in range(0, 50):
	t = threading.Thread(target=send, args=())
	t.start()

