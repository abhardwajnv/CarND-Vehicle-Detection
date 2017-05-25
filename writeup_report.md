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
[image1]: ./output_images/img_vis.png
[image2]: ./output_images/hog_feature.png
[image3]: ./output_images/detect1.png
[image4]: ./output_images/window1.png
[image5]: ./output_images/window2.png
[image6]: ./output_images/window3.png
[image7]: ./output_images/window4.png
[image8]: ./output_images/window_detect.png
[image9]: ./output_images/heatmap.png
[image10]: ./output_images/threshold.png
[image11]: ./output_images/label.png
[image12]: ./output_images/blocks.png
[image13]: ./output_images/test_pipeline.png

[video1]: ./project_video_out.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/abhardwajnv/CarND-Vehicle-Detection/blob/master/writeup_template.md) is the writeup for this project.  

You're reading it!

---

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in 1st to 5th code cells of the IPython notebook located in root folder `./Vehicle_Detection.ipynb`.

The dataset was available into 2 different classes called `vehicle` & `not-vehicle`, as the name suggests `vehicle` class includes all the images of car objects and the `not-vehicle` class includes the non car objects. So i started by loading all of the images from both the classes in individual lists.

Following is the count of images detected:
Car Images      - 8792
Not Car Images  - 8968

Below is the image which shows 16 samples each from both the classes.

![][image1]

To get HOG features i created a method named `get_hog_features` in which i used `skimage.features.hog()` function to extract the HOG features from the dataset and plotted the visualization of them. Below image shows one car image and one non car image along with thier associated histograms of oriented gradients.

![][image2]

Then i created another method named `extract_features` which works with the different color spaces to extract HOG features. This method takes the input lists of image paths defined in earlier section and HOG parameters along with one of a veriety of destination color spaces, which the input image is converted. This gives as an output a flattened array of HOG features for each image from the input list.

Then i defined HOG parameters (`orientation=11`, `pixels_per_cell=16` & `cells_per_block=2`) and fed entire dataset to `extract features` method for feature extraction. These extracted feature sets are combined and respective label vectors are defined ('1' for cars and '0' for not cars) using `numpy.vstack()` & `numpy.hstack()` functions. These features and labels are then shuffled and splitted (using `train_test_split()` function from sklearn library) into training and test sets for training the linear support vector machine (SVM) classifier.

####2. Explain how you settled on your final choice of HOG parameters.

To reach at the final choice of HOG parameters I tried various combinations of parameters and checked the output of SVM classifier. It was based on the accuracy achieved by SVM as well the time taken to achieve that accuracy. I tweaked the parameters until i reached a better accuracy and low enough time with it. I started with low orientations and low pixels per cell rate and gradually increased it with a random amount to finally reach at a stage where i was getting good enough accuracy and training time.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for this section is contained in 6th code cell of IPython notebook.

I trained the linear SVM classifier with HOG features and default classifier parameters with which i was able to achieve the accuracy of 98.37%. I did not use the spatial or channel intensity parameters since i was getting pretty good accuracy with the HOG features alone.

------

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code for this section or the whole pipeline is contained in 7th to 14th code cells.

I started with created a method called `detect()` for detection of car objects in an image. This method combines HOG features extraction with sliding window search, but it does not perform feature extraction on each window individually whereas HOG features are extracted for a specified region in the image. These features are then subsampled according to the size of the window defined and then provided as an input to the classifier. This method then performs the classifier predictions on the HOG featrues for each window region and returns a list of blocks/rectangle objects relative to the window that generated a car prediction. Below image shows output of  `detect()` on one of the test images, using a single window size:

![][image3]

i worked with several different configurations of window sizes and postitions on a frame, with various overlaps in X/Y directions. The following images shows the configurations for all these search windows where windows sizes varied from 1x, 1.5x, 2x & 3x.

Window Size = 1x

![][image4]

Window Size = 1.5x

![][image5]

Window Size = 2x

![][image6]

Window Size = 3x

![][image7]

Trying out on different window sizes was a trade off between all size car detection and false positives where low window size helped detection of small size vehicles but increased the false postitive count whereas keeping the window size high helped in reducing the false positives in the image but not allowing the small visible cars to be detected. So i kept the sizes at a medium range to trade off between both the factors. In addition increasing the overlap in both X & Y directions also helped in increasing the accuracy the detection of actual vehicles and reducing the false positives.

Below image shows the blocks returned by the method which are drawn on one of the test image. You can clearly see the positive detections on the cars visible in the image.

![][image8]

Since in an actual detection of car there are several blocks detected whereas in a false postitive there are ideally one or two blocks detected hence it was required to differentiate between the blocks to draw a single block on a car. For this a combined heatmap `add_heat` and threshold `apply_threshold` is used to achive the same. Heatmap function increments the pixel value of a black region in the image and more overlapping block areas were assigned with more heat. Below image shows the heatmap from the detection on the test image.

![][image9]

A threshold of value 1 is applied the heatmap which sets all the pixels those does not exceed the threshold to 0.
Below image shows the threshold applied on the test image.

![][image10]

Next, scipy's `label()` function was applied on the thresholded image which collects the spatially contiguous areas of the heatmap and assigns each of them with a label. Below image shows the test image applied with label.

![][image11]

This final detected region is then applied with the final block.

![][image12]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

To test the pipeline efficiency i ran all the test images through the pipeline and it was able to detect all the vehicles in the images properly and draw bounding boxes on them. Below image shows the output:

![][image13]

As i stated in earlier questions i mostly focussed on improving the SVM accuracy to get the detection work effectively.
Initially i started with Y color channel with HOG parameters (orientation=8, pixels_per_cell=8) where i was getting low accuracy of about 95%. I changed the color channel to YUV and got the accuracy increated to 97% but the train time got increased with it. Later i increased the HOG parameters to (orientation=11, pixels_per_cell=16) which decreased the train time as well retaining the accuracy close to 98%.

Partially i worked with heatmap/threshold/label steps of improvements as well but i was not getting desired output with that so stopped working on that. Had discussed with few fellow students as well who said my pipeline can be improved with the usage low threshold values on the heatmap. Will try this pipeline as well in future on this project.

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_out.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

Having too many false positives with the default pipeline this was the most importent thing which was required the make the pipeline more robust. I started with creating the heatmap and then applying threshold to the heatmap to locate the positions of the actual vehicles. And then applying scipy's `label()` function to detect the individual blobs from the heatmap to find vehicles and bouding these blobs with boxes. This was still serving with false postives and inaccurate boxes while tracking vehicles. So instead the detections for the past 15 frames were combined and added to the heatmap and the threshold for the heatmap was set to `1 + len(det.prevr)//2` which was performing much better than earlier.

To do this i wrote the method `process_frames()` and a class called `Detections` which stored the previous frames data provided by process_frames() method. Code for these parts is included in 21st and 22nd code cells of IPython notebook.

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Well, this has been the toughest and time consuming project so far. Most of the problems i faced for setting the proper HOG parameters to identify the exact values which should provide me with a best classifier accuracy keeping the training time low enough. After testing different values and combinations of parameters i finalized on the final pipeline numbers which provided me a good enough result for this point.

Another challenge i faced while finalizing on the window size and overllaping of the sliding windows to have accurate detection of the vehicles and reducing the false positives detection. Though my current pipeline still shows a few false positives which has been reduced very much from the initial pipeline.

There are still few issues with the current pipeline.
1. There are still few false positives in the project video.
2. Small size car objects in far view does not get detected.
3. Labels getting lost while tracking fast moving vehicles in the video.
4. Cars coming towards the camera are not detected effectively.
5. For new video's this pipeline might fail if there will be different colors of cars or background.

Following are the improvements i would like to make in future.
1. Make the pipeline more robust to detect all sizes of objects by improving the window sizes and overlapping.
2. Training the classifier with new images to make the pipeline detect different colors of objects.
3. Combining Lane Detection and Object detection together to make a complete pipeline.
