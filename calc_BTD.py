import numpy as np

def bt_ch07_temp_conv(var):
    fk1 = 2.00774e05
    fk2 = 3.68909e03
    bc1 = 0.50777
    bc2 = 0.99929

    return (fk2/(np.log((fk1/var)+1))-bc1)/bc2 - 273.15

def bt_ch14_temp_conv(var):
    fk1 = 8.48310e03
    fk2 = 1.28490e03
    bc1 = 0.25361
    bc2 = 0.9991

    return (fk2/(np.log((fk1/var)+1))-bc1)/bc2 - 273.15

def find_bt_temp_conv(ch):
    if ch == 7:
        return bt_ch07_temp_conv
    else:
        return bt_ch14_temp_conv

# Always returns second_ch - first_ch
def main_func(rad_1, rad_2, first_ch, second_ch):
    func1 = find_bt_temp_conv(first_ch)
    func2 = find_bt_temp_conv(second_ch)

    return func2(rad_2) - func1(rad_1)