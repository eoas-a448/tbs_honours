import argparse
from datetime import datetime, timedelta 
import os
import time
import GOES

parser = argparse.ArgumentParser()
parser.add_argument("start", help="YYYYMMDD")
parser.add_argument("end", help="YYYYMMDD")
parser.add_argument("sat", help="Either goes17 or goes16")
parser.add_argument("data_dir")
args = parser.parse_args()

# start_time = datetime.strptime(args.start, '%y%m%d-%H%M%S')
start_time = datetime.strptime(args.start, '%y%m%d')
current_time = start_time
end_time = datetime.strptime(args.end, '%y%m%d')
data_dir = args.data_dir # Assumed to be top level dir
sat = args.sat

# All downloads will be stored here
root_dir = os.path.join(data_dir, "input/")
os.makedirs(root_dir, exist_ok=True)

# ready.txt system must be made in main tracking script
# it will be generated for one particular day
while current_time <= end_time: #TODO: is <= correct or < ?
    # TODO: Proccess will be astral str -> datetime obj -> downloads str (on the hours)
    sunrise_time = ...
    sunset_time = ...

    sunrise_time_str = ...
    sunset_time_str = ...

    current_time_str = ... # From current_time obv

    todays_dir = os.path.join(root_dir, current_time_str)
    ch2_dir = os.path.join(todays_dir, "ch2")
    ch6_dir = os.path.join(todays_dir, "ch6")
    ch7_dir = os.path.join(todays_dir, "ch7")
    ch14_dir = os.path.join(todays_dir, "ch14")

    os.makedirs(ch2_dir)
    os.makedirs(ch6_dir)
    os.makedirs(ch7_dir)
    os.makedirs(ch14_dir)
    
    # TODO: date string format should be like '20190424-160000'
    # Default should be goes17
    GOES.download(sat, 'ABI-L1b-RadF', channel = ['02'],
              DateTimeIni = sunrise_time_str, DateTimeFin = sunset_time_str,
              rename_fmt = '%Y%m%d%H%M', path_out = ch2_dir)
    GOES.download(sat, 'ABI-L1b-RadF', channel = ['06'],
              DateTimeIni = sunrise_time_str, DateTimeFin = sunset_time_str,
              rename_fmt = '%Y%m%d%H%M', path_out = ch6_dir)
    GOES.download(sat, 'ABI-L1b-RadF', channel = ['07'],
              DateTimeIni = sunrise_time_str, DateTimeFin = sunset_time_str,
              rename_fmt = '%Y%m%d%H%M', path_out = ch7_dir)
    GOES.download(sat, 'ABI-L1b-RadF', channel = ['14'],
              DateTimeIni = sunrise_time_str, DateTimeFin = sunset_time_str,
              rename_fmt = '%Y%m%d%H%M', path_out = ch14_dir)

    # Creates an empty file for signalling
    empty_file = os.path.join(todays_dir, "ready.txt")
    with open(empty_file, 'w') as f: 
        pass

    
    # Add object calls here!!!!! <---------------
    current_time = current_time + timedelta(days = 1)