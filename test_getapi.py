import requests
from datetime import datetime, timedelta

BASE_URL = "http://127.0.0.1:8000"

def test_location_get():
    location_id = "3420004203"  # DB에 있는 실제 location_id
    response = requests.get(f"{BASE_URL}/locations/{location_id}")
    
    print("Status Code:", response.status_code)
    print("Response Text:", response.text)
    
    try:
        print("Location Response:", response.json())
    except requests.exceptions.JSONDecodeError as e:
        print("JSON Decode Error:", e)

def test_link_get():
    # link_id는 실제 DB에 있는 ID로 변경해야 합니다
    link_id = "1760122600"
    response = requests.get(f"{BASE_URL}/links/{link_id}")
    print("Link Response:", response.json())

def test_traffic_get():
    end_time = datetime(2024, 1, 3, 5, 40)
    start_time = end_time - timedelta(hours=1)  # 1시간 전 데이터부터
    
    params = {
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'link_id': '1000026900'  # 선택사항, 실제 link_id로 변경하거나 제거
    }
    
    response = requests.get(f"{BASE_URL}/traffic/", params=params)
    print("Traffic Response:", response.json())


if __name__ == "__main__":
    # print("Testing Location API...")
    # test_location_get()

    # print("\nTesting Link API...")
    # test_link_get()
    #
    print("\nTesting Traffic API...")
    test_traffic_get()