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

Hough transform of a hough transform?