##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

This is it.

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I used a jupyter notebook to carry out this project.

Using the code in cell 1 I was able to do a visual exploration of the car vs not_cars dataset. There were a total of 8792 car images vs 8968 not_car images. Here's a random pick of a car/not_car:

![alt text](./output_images/car_not_car.png "")

I then proceeded to explore the car image by using the HOG transform, an implementation of which is in 'skimage.hog()'. I picked 8 images at random and displayed their hog image next to them to get a feel of what they look like. I chose the Luma/chroma YCrCb space and `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

Here is an example of the output:

![alt text](./output_images/hog_images.png "")

The code that performs this visualization is in cell 2

####2. Explain how you settled on your final choice of HOG parameters.

In cell 2:22-24 one can set the parameters for the hog image features. I experimeted with a few values. Lower values of orientation counts provide more 'generalization' but less detail.
Here is an example with 16 orientations, and `pixels_per_cell=(16,16)`

![alt text](./output_images/hog_images_16_16.png "")

At this level of `pixels_per_cell` it seems gradient detection is a bit too sparse. I ended using the values in the example earlier of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I also used spatial binning and the color histogram to augment the hog features before training the classifer. This can be seen in cell 3. The function `extract_block_features` performs all the feature extractions by calling `bin_spatial` cell 3:61, `color_hist` cell 3:66 and `get_hog_features` on cell 3:78. 

The feature extraction for the dataset if finally performed in `extract_features` on cell 3:103-112. The features are then normalized to make sure they are presented with the same magnitude to the classifer. This is done using `StandardScaler()` in cell 3:120-122. I then display the pre normalized vs normalized features with a histogram to make sure data makes sense. This is an example:

![alt text](./output_images/normalized_features.png "")

The final set of parameters used is defined in the `feat_pars` dic in cell 3:87-99 as follows:
```python
feat_pars = {
    'color_space' : 'YCrCb', # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    'orient' : 9,  # HOG orientations
    'pix_per_cell' : 8, # HOG pixels per cell
    'cell_per_block' : 2, # HOG cells per block
    'hog_channel' : 'ALL', # Can be 0, 1, 2, or "ALL"
    'spatial_size' : (32, 32), # Spatial binning dimensions
    'hist_bins' : 32,    # Number of histogram bins
    'spatial_feat' : True, # Spatial features on or off
    'hist_feat' : True, # Histogram features on or off
    'hist_range' : (0,256), #hist range
    'hog_feat' : True # HOG features on or off
}
```
Hog features are being used in all channels, as well as spatial features and histogram.

As a classifier I used linear SVM using sklearn.svm. The code to train the classifier is seen in cell 4:8-31. I used a 80/20 split for my training/testing sets cell 4:13. 

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

