**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Output_Images/Data_Exploration_result.png "Visualization"
[image2]: ./Output_Images/Pre_Processing_Result.png "PreProcessing"
[image3]: ./new_images/Image1.png "Random Noise"
[image4]: ./new_images/Image2.png "Traffic Sign 1"
[image5]: ./new_images/Image3.png "Traffic Sign 2"
[image6]: ./new_images/Image4.png "Traffic Sign 3"
[image7]: ./new_images/Image5.png "Traffic Sign 4"
[image8]: ./new_images/Image6.png "Traffic Sign 5"
[image9]: ./Output_Images/Softmax_TK2_Values.png "Visualization"
[image10]: ./Output_Images/Guess_Success_Softmax.png "Visualization"

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3) corresponding to RGB images
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.
This is done in lines 67-77. 

Here is an exploratory visualization of the data set. It shows a collection of random traffic signs and the corresponding labels that they have. Note that the labels can be attached to traffic stop sign names using the signames.csv file that it apart of the data set.


![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)
This is done in lines 81-93. 
As a first step, I decided to convert the images to grayscale because the CNN was able to learn from analyzing the shapes. I also normalized the data because it improves efficency processing images. 

Here is an example of a traffic sign image before and after grayscaling/normalzing.

![alt text][image2]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.
My architecture can be found in lines 107-157. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, outputs 28x28x12 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x12 				|
| Convolution 3x3	    | 1x1 stride, outputs 10x10x25      									|
| RELU   |             |                                    
| Max Pooling        | 2x2 stride, outputs 5x5x25          |
| Flatten 		| outputs 625        									|
| Dropout				| outputs 625        									|
| Linear						|		outputs 300										|
|	Linear					|		Outputs 100										|
| Linear     | outputs 43           |
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The Training pipeline can be found in lines 190-221. Note that placeholder variables are elsewhere (160-164). 

To train the model, I used the AdamOptimizer with a learning rate of .001, batch size of 64, and a total of 15 epochs. These were tested and tunned to get the best results. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.
 Printing the results and calling the 'eval' function can be found in 256-267. Note that this also includes evaluating the new road signs that were downloaded from the website. 
 
My final model results were:
* training set accuracy of 97.7%
* validation set accuracy of 95.3%
* test set accuracy of 94.3%

I decided to go with a well-known architecure model by using the LeNet model. I believed it would be good for several reasons. One of them being the LeNet architecture is known to be good for handling image processing. Another is the fact that we learned this method during our lessons so I was more familiar with it and understood more about what was happening behind the scenes of the model. I did apply the dropout method to the model, which slightly improved the results. 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![alt text][image3] ![alt text][image4] ![alt text][image5] 
![alt text][image6] ![alt text][image7] ![alt text][image8]

In general, I picked these images due to them being on the more depective, clearer side of things. I expect my model should do a good job of classifying these specific images that I selected. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The results of analyzing the new images was 100% for the 6 test images that I looked at. This is better than the results of the test set, but this is most likely due to the specific images that were selected. 


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)
this can be found in lines 271-293.

The result of the softmax TK2 applied to the new test images can be seen in the image below:



![alt text][image9]



The guess success rate can be seen in the following image:

![alt text][image10]



As can be seen, the model was able to correctly identify the images that it was given. Like I described earlier, this could be due to the easier images that were used as the test images. 




