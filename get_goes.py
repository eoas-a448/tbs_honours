import GOES

DATA_DIR = "/Users/tschmidt/repos/tgs_honours/good_data/16-ch2-aug08-SHORT/" # This one needs trailing / for goes lib
DATA_DIR1 = "/Users/tschmidt/repos/tgs_honours/good_data/17-aerosols-apr24/" # This one needs trailing / for goes lib
DATA_DIR2 = "/Users/tschmidt/repos/tgs_honours/good_data/17-ch14-aug08/" # This one needs trailing / for goes lib


# GOES.download('goes17', 'ABI-L1b-RadF', channel = ['06'],
#               DateTimeIni = '20190424-160000', DateTimeFin = '20190425-010000',
#               rename_fmt = '%Y%m%d%H%M', path_out = DATA_DIR1)

GOES.download('goes17', 'ABI-L2-ADPF',
              DateTimeIni = '20190424-160000', DateTimeFin = '20190425-010000',
              rename_fmt = '%Y%m%d%H%M', path_out = DATA_DIR1)

####### VISIBLE CHANNEL #########

# GOES.download('goes17', 'ABI-L2-MCMIPF', Channel = ['01'],
#               DateTimeIni = '20190424-160000', DateTimeFin = '20190424-170000',
#               Rename_fmt = '%Y%m%d%H%M', PathOut = DATA_DIR)

# GOES.download('goes16', 'ABI-L2-MCMIPF', Channel = ['01'],
#               DateTimeIni = '20190513-000000', DateTimeFin = '20190513-010000',
#               Rename_fmt = '%Y%m%d%H%M', PathOut = DATA_DIR)