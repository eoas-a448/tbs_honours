Currently getting information about california fire smoke concentrations from 2017-present... <br>
After consulting members of the WFRT, I am likely to choose 2019 as my first year of analysis. <br>

I need to determine the exact type of weather event that makes the ship tracks more obvious. (The kind of event that took place off of the coast of California on April 24-25 2019) <br>

Kalman filter used becuase ship tracks have different velocity than surrounding clouds? <br>

New event now being studied. (Sept 12 14:00 to Sept 13 03:00) <br>

Originally, I was hoping to use a kalman filter to assist with the classification/clustering process, but the more I study the Kalman filter the more I have trouble seeing how it can be used directly. Usually Kalman filters are used for object tracking, but these objects are usually located and classified before being fed into the filter since the filter needs positional and some sort of "velocity" information. (Image segmentation) There seems to be very little literature on using a Kalman filter for the classification step itself. <br>
A different route that I am current trying is to try a Canny filter to detect edges in our data (similar to the sobel operator). I am hoping to use this in combination with a variety of threshold filters to isolate the structures we often associate with ship tracks. I am trying to use the Canny filter here instead of the more common Sobel silter since it seems the gassian filter is built into the Canny filter so I hope to save some runtime with it. <br>

Is the kernel standard deviation that Phil mentioned related to gaussian smoothing/filtering? It does not seem so. <br>

I tried to filter out any data with a BT higher than a fixed threshold. Unlike the high-cloud filter, this seemed to fail. The temperatures of the ship tracks seem to vary too much between examples. <br>

I have now expanded my proccess to include both an edge detection step and a following straight-line detection step. The concept being that the straighter edges (after various data filtering steps) would indicate ship tracks with an acceptable level of error. As of today (Nov.12) I have tried Canny edge detection, a Hessian filter, and a non-local mean denoising filter for the edge detection step. For the line detection I have used a Probablistic Hough Transform. So far I have had mixed results. In relativly ideal scenarios (lots of ship tracks, low noise) the canny edge detection and probablistic Hough transform combination seems to work fairly well, but in non-ideal situations it seems to fail quite badly. The most obvious solution at this point is to use more data filtering steps where "obvious non shiptrack structures" such as non-cloud data is removed beforehand. This is still being discussed with phil. <br>

Hough transform of a hough transform? <br>

Check if openstreetmaps is copyrighted \<---------------- <br>

Object tracking can be either with the canny edges or with the original BTD data <br>

TODO: Figure out if golden arches works at night. **RESPONSE:** Phil recommends not trying night stuff for now. I agree.<br>

TODO: Find better way of labelling the tracks within each box <br>

TODO: Fix golden arches!!!!!!!!!! <br>
It seems that the golden arches algorithm is primarily meant for open ocean. <br>

TODO: See if we can use the conus disk for goes-17 data <br>

TODO: Test what happens when I display full dataset for goes-17 data. See if black area still exists. **RESPONSE:** It seems to be a problem with the GOES library slice/get_lonlat functions. The data itself is fine. This means I need to figure out an alternative way of generating latlon meshes and slicing the data to a bounding box. <br>

**MAJOR UPDATE JAN.27:** After numerous problems with the golden arches strat I found that the idea Phil proposed of comparing shortwave reflectivity to longwave brightness temperature has worked very well (much much better than golden arches). It seems to filter out the exact areas of the Apr.24th 2019 data I was hoping an open-ocean filter would. I am currently using kmeans to find the two clusters, but I may try more effective clustering methods later. Now it just has to be tested across more days to make sure it is truly robust. <br>

NOTE: As an aside I tried the golden arch algorithm on the LR->HR resampled data to see if it worked any better. This did not seem to be the case. It did not work for either the full dataset strategy nor for the "sliced" dataset strategy I used to try to emulate the resulsts of the golden arch paper more closely. <br>

TODO: Fix the issue where the resampled LR longwave data has too many parallel "stretching" lines. This is bad because it is making hough lines and tracking nearly impossible. One possible solution is to resample back to the LR dimensions after all the filtering has occured. This or I will have to resample the HR shortwave data down to the LR dimensions during the actual filtering step (lose some defintion) but I want to avoid doing this as much as possible. **Try gaussian filter with sigma of 0.8X4???? (Becuase 4 times old sigma for stretch factor)**