import argparse
import rasterio
from cv2 import cv2
from datetime import datetime
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("date", help="YYYYMMDD")
parser.add_argument("data_dir")
args = parser.parse_args()

date = args.date
data_dir = args.data_dir

true_label_dir = os.path.join(data_dir, "true_labels", date)
output_label_dir = os.path.join(data_dir, "output", date, "labels")

####TEMP#######################
true_label_dir = "/Users/tschmidt/repos/tgs_honours/output/apr_24_open_ocean/true_label/"
output_label_dir = "/Users/tschmidt/repos/tgs_honours/output/apr_24_open_ocean/output/"
##############################

# Get contents of true label dir
true_label_list = os.listdir(true_label_dir)
if ".DS_Store" in true_label_list:
    true_label_list.remove(".DS_Store") # For mac users
true_label_list = sorted(true_label_list)

# Get contents of output label dir
output_label_list = os.listdir(output_label_dir)
if ".DS_Store" in output_label_list:
    output_label_list.remove(".DS_Store") # For mac users
output_label_list = sorted(output_label_list)

true_pos_rates = []
true_neg_rates = []
false_pos_rates = []
false_neg_rates = []
i = 0
while i < len(true_label_list):
    output_name = output_label_list[i]
    act_name = true_label_list[i]
    
    output_label_path = os.path.join(output_label_dir, output_name)
    true_label_path = os.path.join(true_label_dir, act_name)

    # true_label = cv2.imread(true_label_path)[:,:,0] # Just have to use one band becuase it is white

    ###TEMP##########################################
    src = rasterio.open(true_label_path, mode='r+')
    true_label = src.read(1)
    src.close() # Free memory
    src = None
    #############################################

    src = rasterio.open(output_label_path, mode='r+')
    output_label = src.read(1)
    src.close() # Free memory
    src = None

    # Calculate true positive rate
    positive = len(true_label[true_label != 0])
    true_pos_rate = np.sum(output_label[true_label != 0])/positive

    # Calculate the true negative rate
    negative = len(true_label[true_label == 0])
    true_neg_rate = (-1)*np.sum(output_label[true_label == 0]-1)/negative

    # Calculate the false positive rate
    false_pos_rate = 1 - true_neg_rate

    # Calculate the false negative rate
    false_neg_rate = 1 - true_pos_rate

    true_pos_rates.append(true_pos_rate)
    true_neg_rates.append(true_neg_rate)
    false_pos_rates.append(false_pos_rate)
    false_neg_rates.append(false_neg_rate)

    i = i + 1

print("True pos rate= " + str(np.mean(true_pos_rates)))
print("True neg rate= " + str(np.mean(true_neg_rates)))
print("False pos rate= " + str(np.mean(false_pos_rates)))
print("False neg rate= " + str(np.mean(false_neg_rates)))