# Vehicle Detection
### Project 5, Term 1, Udacity CarND
#### Jon Wetherbee, Sept 2017 Cohort

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector. 
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream and create a heat map of recurring detections, frame by frame, to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected, overlay this for each original frame, and generate a new annotated video.

[//]: # (Image References)
[image1]: ./output_images/cars_not_cars.png
[image2]: ./output_images/hog_features.png
[image3]: ./examples/sliding_windows.jpg
[image4a]: ./output_images/just_boxes.png
[image4b]: ./output_images/just_boxes2.png
[image5a]: ./output_images/test1.png
[image5b]: ./output_images/test2.png
[image5c]: ./output_images/test3.png
[image5d]: ./output_images/test4.png
[image5e]: ./output_images/test5.png
[image5f]: ./output_images/test6.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

### Here are the [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

What follows is a description of how I addressed each point in my implementation. 

**All code references are to the Jupiter notebook I wrote for this project: [CarND-Vehicle-Detection4.ipynb](./CarND-Vehicle-Detection4.ipynb)**.

---

## Histogram of Oriented Gradients (HOG)

###  Extracting HOG features from the training images

#### Loading the Training Data

In *Code Section 1: Load the training data, lines 18-29*, I began by reading in all the `vehicle` and `non-vehicle` images:

```python
# Divide up into cars and notcars
images = glob.glob('vehicles/**/*.png', recursive=True)
cars = []
for image in images:
    cars.append(image)
print('# of cars:   ',len(cars))

images = glob.glob('non-vehicles/**/*.png', recursive=True)
notcars = []
for image in images:
    notcars.append(image)
print('# of notcars:',len(notcars))
```

#### Displaying Examples from the Training Dataset

In *Code section 2: Display examples from the training dataset*, I show some examples of both the `vehicle` and `non-vehicle` images:

![alt text][image1]

#### Feature Extraction Methods

*Code section 3: Feature extraction methods* contains methods for applying the techniques we covered in the lessons:

1. HOG (Histogram of Oriented Gradient)
2. Binary Spatial data
3. Color histogram analysis

#### Displaying HOG (Histogram of Oriented Gradients) Representations of Training Images

The first technique, HOG, is implemented in *Code section 4: Display HOG (Histogram of Oriented Gradients) representations of training images*.

In this code block I show examples of both cars and non-cars, experimenting with different colors and showing the HOG results for each color channel. The colors I tested are:

1. BGR
2. LUV
3. YUV
4. YCrCb

and the HOG parameters I chose were:

```python
orient = 11 
pix_per_cell = 8 
cell_per_block = 2
```

The calls to `skimage.hog()` in my `get_hog_features()` method, lines 14-18, are here:

```python
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec)
```

I conclude the code block by grabbing four random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like:

![alt text][image2]

### Choosing HOG Parameters

To better understand the HOG feature detection technique in isolation, I tried various combinations of parameters and ran them through my pipeline to process first a handful of example images, and then two videos supplied with the project. My aim was to decrease both:

1. False-negatives, cases where cars were present in the image, but not detected
2. False-positives, when the classifier detected a car where there was none

I experimented with different values for the three parameters above, and different color transforms, until I arrived at results that were able to pretty reliably detect the cars in sample images and ignore everything else.

The next sections describe how I fleshed out my pipeline so that I could see the results of my experimentation, including by use of two other feature detection techniques: binary spatial, and color histogram.

### Extracting HOG and color features

In *Code section 5: Methods for extracting features from images*, for each image in the training classes, I extracted its HOG, binary spatial, and color histogram features into a single feature vector. A corresponding labels vector was created for each, with value 1 for each car image, and 0 for each non-car. 

In *Code section 6: Extract features from the training data, lines 21-24*, to normalize the results of the different feature detectors, I fit each feature vector using a StandardScalar():

```python
# Fit a per-column scaler - this will be necessary if combining different types of features (HOG + color_hist/bin_spatial)
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)
```

I split this normalized training data into training and test sets, and then was able to train my SVC classifier.

### Training a classifier using my selected HOG features and color features

In *Code section 7: Train and validate a LinearSVC* I began by creating an `sklearn.svm.LinearSVC` (lines 1-2):

```python
# Use a linear SVC 
svc = LinearSVC()
```

fitted the classifier with the training data (line 6):

```python
svc.fit(X_train, y_train)
```

and then validated the resulting classifier using the test set(line 17):

```python
print('My SVC predicts:     ', svc.predict(X_test[0:n_predict]))
```

The results of the validation (run on my MacBook) are shown here:

```python
5.74 Seconds to train SVC...
Test Accuracy of SVC =  0.9935
My SVC predicts:      [ 1.  1.  0.  1.  0.  1.  0.  1.  0.  0.]
For these 10 labels:  [ 1.  1.  0.  1.  0.  1.  0.  1.  0.  0.]
0.00166 Seconds to predict 10 labels with SVC
```

## Detecting Cars in Images

Armed with a trained classifier, it was now possible to write the code which would use this classifier to detect cars in images, and demonstrate the detection by drawing boxes around the cars that were found. 

The classifier was trained to identify cars in images of size 64x64 pixels, but the sample images it will be processing are potentially much larger. To handle this, I set up a strategy of parceling the image into sub-sections, and applying the classifier to one sub-section at a time. This strategy is known as sliding window search.

### Sliding Window Search Technique

The gist of this approach is we choose a sub-section window size, and then starting at the top left of the sample image, we apply feature detection to this first sub-section of the image. Thinking of this sub-section as a "window" into the larger sample image, we then slide this window over a few pixels, and apply the feature detection to this new sub-section of the sample image. We continue this until we hit the the right side of the image and can't slide over any further. At this point we return to the left side, drop down a few pixels, and continue sliding to the right as far as we can again, detecting features in each sub-section of the sample image as we go. This continues until we arrive at the bottom right corner of the image. We have effectively scanned overlapping sub-sections of the entire original source image. (Well, with one caveat: we cropped the sample image to be begin with, so that we were only scanning the portion that could potentially contain cars. Eliminating unnecessary processing speeds up the detection process.)

The code for all this is in *Code section 8: Method for detecting and locating features using sliding windows, lines 21-87*, in the method `find_cars()'.

### Scaling the sample image

Furthermore, because cars can appear in varying sizes within the sample image, I employed a complementary strategy of "scaling" the source image to reduce its resolution, and then applying the sliding windows to these scaled renditions. With the correct scaling levels, this allows the classifier to find cars of all different sizes within the sample image.

*Code section 12: Image annotation using scaled search, lines 5-8*, demonstrates how this simple but powerful scaling strategy is invoked:

```python
    for scale in scale_list:
        rects = find_cars(img, ystart, ystop, scale, 
                          svc, X_scaler, orient, pix_per_cell, cell_per_block)
        rectangles.extend(rects)
```

Inside the find_cars() method (*Code section 8: Method for detecting and locating features using sliding windows, lines 31-33*), this 'scale' param is used to scale down the resolution of the image prior to performing the sliding box technique:

```python
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
```

I experimented with various scale_list values, seeking a balance between the benefit of being able to detect a wider range of car sizes vs. the performance overhead of additional feature extraction processing, and ended up with this:

```python
scale_list = [1, 1.5, 2, 3]
```

### Examples of Annotated Images

The following example images were generated using the four scales, above, and using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector:

![alt text][image4a]    ![alt text][image4b]

---

## Video Implementation

### Here's a [link to my video result](./output_images/project_video_out_final.mp4)

### Generating Smooth, Contiguous Results with a  Minimum of False Positives

When we run our classifier on an image using the sliding scaled boxes technique above, we produce a set of rectangles that indicate the location of cars on our sample image. This is useful, but we can improve the output by further processing this info and producing a single box for each car. We can also make an effort to ensure that the location of that box remains fairly consistent from one frame to the next, matching the consistency of the car's appearance in the frames.

There are two main techniques I employ to produce this result:

1. Use a thresholded heat map to reduce a cluster of car detections into a single rectangle on the image
2. Introduce a CarHistory class to track the car locations identified over the last 10 (or some variable number of) frames

#### Using a Heat Map

In *Code section 11: Heatmap, threshold, and box-drawing utility methods* I list a handful of methods (from the lessons) for tracking the concentration of detected cars. The results are then normalized and assigned labels, using the `scipy.ndimage.measurements.label()` method, in *Code section 12: Image annotation using scaled search, lines 33-37*:

```python
    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
```

The result of this is used to produce a graphical heat map that will be shown in a moment, below.

#### CarHistory Class

When processing a sequence of images, as from a video, we can benefit from the knowledge that cars travel in a (fairly!) consistent way along the road. For the purposes of this project, this means we can assume that when a car is detected in one frame, it is probably going to appear in a similar location in the next frame as well, unless it is on the edge of the frame. To take advantage of this knowledge, I created a class to track the frame history, recording all of the identified cars for the previous 10 (or some variable number) of frames.

This class is found in *Code section 10: CarHistory class: Record cars found in recent frames*, and I documented it as:

```python
# The CarHistory class holds the rectangles detected by find_cars
# for the previous n video frames. These rectangles are combined
# and the sum is passed to the heat map detection and thresholding
# methods.
#
# This history is also used to ignore spurious false-positive cars 
# that appear only in one or two frames.
```

#### Combination of the two

Combining the CarHistory class with heat map processing helps to smooth out our car detection results in three ways:

1. When the classifier fails to detect a car in one frame, but it was detected in some number of previous frames, by taking the sum of all detections over the past 10 frames we can still decide, after thresholding, to display the box for this car.
2. Averaging the detections over a series of frames, and applying a threshold mask, can also filter out false-positives. Spurious detections appearing on one or two frames, but absent from most of the previous 10, will get filtered out by the thresholding mechanism.
3. By sending the previous 10 frames' detections to the heat map processor each time we process a frame, the boxes become very consistent from one frame to the next, giving a smooth appearance.

Returning to *Code section 12: Image annotation using scaled search, lines 18-24*, here is how the CarHistory class is used to capture the detection history and serve it up when generating the heat map:

```python
        carHistory.add_cars(rectangles)

        # Add heat to each box in box list
        heat = add_heat(heat, carHistory.get_cars())

        # Apply threshold to help remove false positives
        heat = apply_threshold(heat, carHistory.get_frame_count()//2)
```
	
you can see that I added the rectangles (car locations) detected for the current frame to the CarHistory class. It only ever keeps a max of 10 frames worth of history, aging them out in FIFO order. I then pass the sum of the previous 10 frames' rectangles to the add_heat() method to generate the resulting heat map and a single rectangle that surrounds each heat map blob.

Here's an example result showing the heatmap from a series of individual frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video. In this case the images were processed in isolation, so the CarHistory class was not used. (Note that the CarHistory class was used when processing the frames of the project video referenced above.)

## Six sample frames, transformed and annotated

For each of the sample images below:

* The left image shows the frame with a separate rectangle overlaid for each car instance detected by a feature detector.
* The center image shows the heat map corresponding to the concentration of rectangles.
* The right image shows the final result: a single rectangle surrounding each car that was detected.

![alt text][image5a]
![alt text][image5b]
![alt text][image5c]
![alt text][image5d]
![alt text][image5e]
![alt text][image5f]

---

## Discussion

### Problems / issues faced in my implementation of this project

A reasonable question to ask is, "Where will my pipeline likely fail?"

#### Non-freeway environment

While my pipeline seems to work well for freeway driving, where the only oncoming traffic is in the distance and beyond a guardrail, it might well fail in an urban setting or on a road with no lane divider. The smoothing I added makes assumptions about the relatively constance of surrounding cars from one frame to the next. High-speed oncoming traffic could well negate those assumptions.

#### Performance: Run-time processing

I'm concerned that my pipeline would lag while processing images in real-time. With some parameter settings it over an hour to process the ~1 minute project video. At that rate, it would only be able to munge through a frame every 3-4 seconds, which would not be adequate. Better hardware (I'm on a MacBoo) and greater attention to code and parameter optimization would likely drive the frame processing throughput, but this is an important issue to explore further.

### For further exploration

That said, the areas I would like to pursue further are:

#### Auto-tuning the classifier(s)

Auto-tuning the classifier to detect which:

* scale sizes
* filter thresholds
* colorspace
* pixels_per_cell
* cells_per_block
* orientation
* hog_channel
* spatial_size
* hist_bins
* bin_size 

parameters produce the optimal results feels like a promising path to pursue. I was excited to come across this in the lesson -- GridSearchCV and RandomizedSearchCV -- because it had occurred to me earlier in the term that we could really benefit from tools like this, instead of using trial and error with only a handful of parameter combinations.

#### Reducing the fps

Along the lines of the performance concern I raised above, it seems like we could get away with processing only a fraction of the frames in the video. The surrounding scenery may whiz by, but the cars on the freeway don't move very quickly relative to ours, and they are the only thing we're interested in detecting here. Even if we could process 24fps in real-time, I'm confident we could put our resources to better use, on a task like this, extending the feature set or increasing the granularity of the sliding window search panels, or in some other area.



