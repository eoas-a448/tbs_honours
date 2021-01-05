import argparse
from datetime import datetime, timedelta 
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument("start", help="YYYYMMDD-HHMMSS")
parser.add_argument("end", help="YYYYMMDD-HHMMSS")
parser.add_argument("data_dir")
args = parser.parse_args()

start_time = datetime.strptime(args.start, '%y%m%d-%H%M%S')
current_time = start_time
end_time = datetime.strptime(args.end, '%y%m%d-%H%M%S')
data_dir = args.data_dir

while current_time <= end_time:
    while True:
        current_time_string = current_time.strftime("%y%m%d-%H%M%S")
        current_time_path = os.path.join(data_dir,current_time_string)
        ready_file = os.path.join(current_time_path, "ready.txt")
        if os.path.exists(current_time_path):
            if os.path.exists(ready_file):
                break

        time.sleep(15)

    # Add object calls here!!!!! <---------------
    current_time = current_time + timedelta(days = 1)