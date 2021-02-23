import argparse
from datetime import datetime, timedelta 
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument("start", help="YYYYMMDD")
parser.add_argument("end", help="YYYYMMDD")
parser.add_argument("data_dir")
args = parser.parse_args()

# start_time = datetime.strptime(args.start, '%y%m%d-%H%M%S')
start_time = datetime.strptime(args.start, '%y%m%d')
current_time = start_time
end_time = datetime.strptime(args.end, '%y%m%d')
data_dir = args.data_dir # Assumed to be top level dir

input_dir = os.path.join(data_dir, "input/")
output_dir = os.path.join(data_dir, "output/")
os.makedirs(output_dir, exist_ok=True)

# ready.txt system must be made in main tracking script
# it will be generated for one particular day
while current_time <= end_time:
    while True:
        current_time_string = current_time.strftime("%y%m%d")
        current_time_path = os.path.join(input_dir,current_time_string)
        ready_file = os.path.join(current_time_path, "ready.txt")
        if os.path.exists(current_time_path):
            if os.path.exists(ready_file):
                break

        time.sleep(15)

    output_current_time_path = os.path.join(output_dir, current_time_string)
    os.makedirs(output_current_time_path)
    # Add object calls here!!!!! <---------------
    current_time = current_time + timedelta(days = 1)