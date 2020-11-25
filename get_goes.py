import GOES

# DATA_DIR = "/Users/tschmidt/repos/tgs_honours/good_data/"
DATA_DIR1 = "/Users/tschmidt/repos/tgs_honours/good_data/16-ch7-may13/" # This one needs trailing / for goes lib
DATA_DIR2 = "/Users/tschmidt/repos/tgs_honours/good_data/16-ch14-may13/" # This one needs trailing / for goes lib

GOES.download('goes16', 'ABI-L1b-RadF', Channel = ['07'],
              DateTimeIni = '20190513-000000', DateTimeFin = '20190514-000000',
              Rename_fmt = '%Y%m%d%H%M', PathOut = DATA_DIR1)
GOES.download('goes16', 'ABI-L1b-RadF', Channel = ['14'],
              DateTimeIni = '20190513-000000', DateTimeFin = '20190514-000000',
              Rename_fmt = '%Y%m%d%H%M', PathOut = DATA_DIR2)

# GOES.download('goes16', 'ABI-L2-MCMIPF', Channel = ['01'],
#               DateTimeIni = '20190424-140000', DateTimeFin = '20190424-180000',
#               Rename_fmt = '%Y%m%d%H%M', PathOut = DATA_DIR)