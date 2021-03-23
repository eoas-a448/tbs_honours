import random
from skimage.draw import line
from skimage.draw import rectangle
import numpy as np
from cv2 import cv2
import os
from scipy import ndimage

class ShipTrack():
    points = []
    segment_widths = [] # radius

    def __init__(self, max_vertices, max_seg_len, max_seg_width, img_row_num, img_col_num, is_straight=True):
        self.points = []
        self.segment_widths = []

        vertices_num = random.randint(2, max_vertices)

        self.points.append((random.randrange(img_row_num),random.randrange(img_col_num))) # Find position in scene

        if is_straight:
            vertices_num = 2
            max_seg_len = max_seg_len*3

        for i in range(vertices_num-1):
            last_point = self.points[-1]

            new_row_min = last_point[0] - max_seg_len
            if new_row_min < 0:
                new_row_min = 0
            
            new_row_max = last_point[0] + max_seg_len
            if new_row_max >= img_row_num:
                new_row_max = img_row_num - 1

            new_col_min = last_point[1] - max_seg_len
            if new_col_min < 0:
                new_col_min = 0

            new_col_max = last_point[1] + max_seg_len
            if new_col_max >= img_col_num:
                new_col_max = img_col_num - 1

            self.points.append((random.randint(new_row_min, new_row_max),random.randint(new_col_min, new_col_max)))
            self.segment_widths.append(random.randint(2, max_seg_width))

    
    # Here image is meant to be the track layer
    def draw(self, image, labels, mu, sigma):
        for i in range(len(self.segment_widths)):
            width = self.segment_widths[i]
            line_points = (self.points[i],self.points[i+1])

            line_pixels = list(zip(*line(*line_points[0], *line_points[1])))
            normal_data_image = np.random.normal(loc=mu, scale=sigma, size=image.shape) # This is done because each square may not be (width x width)

            # TODO: make sure this is right way to use pixels
            for pixel in line_pixels:
                rows, cols = rectangle(pixel, extent=(width,width), shape=image.shape)

                image[rows,cols] = normal_data_image[rows,cols]
                labels[rows,cols] = 1


class Layer():
    image = None
    img_row_num = None
    img_col_num = None

    # mu and sigma here are for the backround's values
    def __init__(self, img_row_num, img_col_num):
        self.image = np.zeros((img_row_num, img_col_num))
        self.img_row_num = img_row_num
        self.img_col_num = img_col_num


    def backround_draw(self, mu, sigma):
        self.image = np.random.normal(loc=mu, scale=sigma, size=(self.img_row_num,self.img_col_num))


ROW_LEN = 800
COL_LEN = 800
OUT_DIR = "/Users/tschmidt/repos/tgs_honours/output/"

backround = Layer(ROW_LEN,COL_LEN)
backround.backround_draw(10,4)
labels = Layer(ROW_LEN,COL_LEN)

tracks = []

for i in range(10):
    tracks.append(ShipTrack(5,100,6,ROW_LEN,COL_LEN,False))

for track in tracks:
    track.draw(backround.image, labels.image, 20, 2)

BTD_img = np.float32(backround.image)

# Output image
min_BTD = np.nanmin(BTD_img)
if min_BTD < 0:
    BTD_img = BTD_img + np.abs(min_BTD)
max_BTD = np.nanmax(BTD_img)
BTD_img = BTD_img/max_BTD
BTD_img = cv2.cvtColor(BTD_img*255, cv2.COLOR_GRAY2BGR)

filename = "simulation.png"
file_path = os.path.join(OUT_DIR, filename)
cv2.imwrite(file_path, BTD_img)

n = 40
im = np.zeros((ROW_LEN, COL_LEN))
points = ROW_LEN*np.random.random((2, n**2))
im[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
im = ndimage.gaussian_filter(im, sigma=ROW_LEN/(4.*(n/4)))

im = (im > im.mean()).astype(np.float)

img = np.float32(im)

# Output image
min_BTD = np.nanmin(img)
if min_BTD < 0:
    img = img + np.abs(min_BTD)
max_BTD = np.nanmax(img)
img = img/max_BTD
img = cv2.cvtColor(img*255, cv2.COLOR_GRAY2BGR)

filename = "simulation_blobs.png"
file_path = os.path.join(OUT_DIR, filename)
cv2.imwrite(file_path, img)