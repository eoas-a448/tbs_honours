import GOES

DATA_DIR = "/Users/tschmidt/repos/tgs_honours/good_data/16-ch2-aug08-SHORT/" # This one needs trailing / for goes lib
DATA_DIR1 = "/Users/tschmidt/repos/tgs_honours/good_data/17-ch2-apr24/" # This one needs trailing / for goes lib
DATA_DIR2 = "/Users/tschmidt/repos/tgs_honours/good_data/17-ch14-aug08/" # This one needs trailing / for goes lib

# GOES.download('goes16', 'ABI-L1b-RadF', Channel = ['07'],
#               DateTimeIni = '20190513-000000', DateTimeFin = '20190514-000000',
#               Rename_fmt = '%Y%m%d%H%M', PathOut = DATA_DIR1)
# GOES.download('goes16', 'ABI-L1b-RadF', Channel = ['14'],
#               DateTimeIni = '20190513-000000', DateTimeFin = '20190514-000000',
#               Rename_fmt = '%Y%m%d%H%M', PathOut = DATA_DIR2)

# GOES.download('goes17', 'ABI-L1b-RadF', Channel = ['07'],
#               DateTimeIni = '20190424-140000', DateTimeFin = '20190425-060000',
#               Rename_fmt = '%Y%m%d%H%M', PathOut = DATA_DIR1)
# GOES.download('goes17', 'ABI-L1b-RadF', Channel = ['14'],
#               DateTimeIni = '20190424-140000', DateTimeFin = '20190425-060000',
#               Rename_fmt = '%Y%m%d%H%M', PathOut = DATA_DIR2)

# GOES.download('goes17', 'ABI-L1b-RadF', Channel = ['07'],
#               DateTimeIni = '20190807-220000', DateTimeFin = '20190808-050000',
#               Rename_fmt = '%Y%m%d%H%M', PathOut = DATA_DIR1)
# GOES.download('goes17', 'ABI-L1b-RadF', Channel = ['14'],
#               DateTimeIni = '20190807-220000', DateTimeFin = '20190808-050000',
#               Rename_fmt = '%Y%m%d%H%M', PathOut = DATA_DIR2)

# GOES.download('goes16', 'ABI-L1b-RadF', Channel = ['02'],
#               DateTimeIni = '20190424-160000', DateTimeFin = '20190425-010000',
#               Rename_fmt = '%Y%m%d%H%M', PathOut = DATA_DIR1)

GOES.download('goes17', 'ABI-L1b-RadF', Channel = ['02'],
              DateTimeIni = '20190424-160000', DateTimeFin = '20190425-060000',
              Rename_fmt = '%Y%m%d%H%M', PathOut = DATA_DIR1)

####### VISIBLE CHANNEL #########

# GOES.download('goes17', 'ABI-L2-MCMIPF', Channel = ['01'],
#               DateTimeIni = '20190424-160000', DateTimeFin = '20190424-170000',
#               Rename_fmt = '%Y%m%d%H%M', PathOut = DATA_DIR)

# GOES.download('goes16', 'ABI-L2-MCMIPF', Channel = ['01'],
#               DateTimeIni = '20190513-000000', DateTimeFin = '20190513-010000',
#               Rename_fmt = '%Y%m%d%H%M', PathOut = DATA_DIR)