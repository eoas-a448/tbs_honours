import GOES
from astral import LocationInfo
from astral.sun import sun
from datetime import datetime, timedelta

LA_lon = -118.243683
LA_lat = 34.052235

loc = LocationInfo(latitude=LA_lat, longitude=LA_lon)

DATA_DIR = "/Users/tschmidt/repos/tgs_honours/good_data/16-ch2-aug08-SHORT/" # This one needs trailing / for goes lib
DATA_DIR1 = "/Users/tschmidt/repos/tgs_honours/good_data/17-aerosols-apr24/" # This one needs trailing / for goes lib
DATA_DIR2 = "/Users/tschmidt/repos/tgs_honours/good_data/17-ch2-jul10/" # This one needs trailing / for goes lib


# GOES.download('goes17', 'ABI-L1b-RadF', channel = ['06'],
#               DateTimeIni = '20190424-160000', DateTimeFin = '20190425-010000',
#               rename_fmt = '%Y%m%d%H%M', path_out = DATA_DIR1)

# GOES.download('goes17', 'ABI-L1b-RadF', channel = ['02', '06'],
#               DateTimeIni = '20190807-220000', DateTimeFin = '20190808-050000',
#               rename_fmt = '%Y%m%d%H%M', path_out = DATA_DIR2)

current_time = datetime.strptime("20190710", '%Y%m%d')
s = sun(loc.observer, date=current_time)
sunrise_time = s["sunrise"] + timedelta(hours=3)
sunset_time = s["sunset"] - timedelta(hours=3)
sunrise_time_str = sunrise_time.strftime("%Y%m%d-%H") + "0000"
sunset_time_str = sunset_time.strftime("%Y%m%d-%H") + "0000"
GOES.download('goes17', 'ABI-L1b-RadF', channel = ['02', '06', '07', '14'],
              DateTimeIni = sunrise_time_str, DateTimeFin = sunset_time_str,
              rename_fmt = '%Y%m%d%H%M', path_out = DATA_DIR2)

####### VISIBLE CHANNEL #########

# GOES.download('goes17', 'ABI-L2-MCMIPF', Channel = ['01'],
#               DateTimeIni = '20190424-160000', DateTimeFin = '20190424-170000',
#               Rename_fmt = '%Y%m%d%H%M', PathOut = DATA_DIR)

# GOES.download('goes16', 'ABI-L2-MCMIPF', Channel = ['01'],
#               DateTimeIni = '20190513-000000', DateTimeFin = '20190513-010000',
#               Rename_fmt = '%Y%m%d%H%M', PathOut = DATA_DIR)