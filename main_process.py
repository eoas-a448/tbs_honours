import argparse
from datetime import datetime, timedelta 
import os
import time

from make_pngs import run

parser = argparse.ArgumentParser()
parser.add_argument("start", help="YYYYMMDD")
parser.add_argument("end", help="YYYYMMDD")
parser.add_argument("data_dir")
args = parser.parse_args()

# start_time = datetime.strptime(args.start, '%y%m%d-%H%M%S')
start_time = datetime.strptime(args.start, '%Y%m%d')
current_time = start_time
end_time = datetime.strptime(args.end, '%Y%m%d')
data_dir = args.data_dir # Assumed to be top level dir

input_dir = os.path.join(data_dir, "input/")
output_dir = os.path.join(data_dir, "output/")
os.makedirs(output_dir, exist_ok=True)
label_name = "labels"
data_name = "data"

# ready.txt system must be made in main tracking script
# it will be generated for one particular day
while current_time <= end_time:
    while True:
        current_time_string = current_time.strftime("%Y%m%d")
        current_time_path = os.path.join(input_dir,current_time_string)
        ready_file = os.path.join(current_time_path, "ready.txt")
        if os.path.exists(current_time_path):
            if os.path.exists(ready_file):
                break

        time.sleep(15)

    output_current_time_path = os.path.join(output_dir, current_time_string)
    label_img_path = os.path.join(output_current_time_path, label_name)
    data_img_path = os.path.join(output_current_time_path, data_name)
    os.makedirs(label_img_path)
    os.makedirs(data_img_path)
    
    run(current_time_path, output_current_time_path)

    current_time = current_time + timedelta(days = 1)