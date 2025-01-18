import logging
import warnings
from model import utility as U
        
# 설정 파일 읽기 
def read_config():
   config = U.configparser.ConfigParser()
   config.read('config.ini')
   
   # 읽은 값 출력해보기
   print(f"총 배열 개수: {config['arrays']['number']}")   
   return config

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    y=U.TestScript()
    print(y)
    print(y.shape)
    reader=U.Datareader(option='test')
    A,B,C,D,E,F=reader.testarrays.copy()
    print(f'test data:{A},{B},{C},{D},{E},{F}')
    out=reader.process_data(A,B,C,D,E,F)
    print (f'outdata:{out}')
    print (out.shape)