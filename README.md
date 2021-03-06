**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images


[//]: # (Image References)

[image1]: ./Output_Images/Data_Exploration_result.png "Visualization"
[image2]: ./Output_Images/Pre_Processing_Result.png "PreProcessing"
[image3]: ./new_images/Image1.png "Random Noise"
[image4]: ./new_images/Image2.png "Traffic Sign 1"
[image5]: ./new_images/Image3.png "Traffic Sign 2"
[image6]: ./new_images/Image4.png "Traffic Sign 3"
[image7]: ./new_images/Image5.png "Traffic Sign 4"
[image8]: ./new_images/Image6.png "Traffic Sign 5"
[image10]: ./Output_Images/Guess_Success_Softmax.png "Visualization"

---
### Writeup / README

### Data Set Summary & Exploration
A link to the dataset that was used for this project: [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3) corresponding to RGB images
* The number of unique classes/labels in the data set is 43

Here are multiple images that depict the type of data that is in the dataset. It shows a collection of random traffic signs and the corresponding labels that they have. Note that the labels can be attached to traffic stop sign names using the signames.csv file that it apart of the data set.


![alt text][image1]

### Design and Test a Model Architecture

As a first step, I decided to convert the images to grayscale. After reading a multitude of studies about image processing and using CCN', it seemed like the best approach. I also ended up using the dropout method which ended up improving the accuracy of the algorithm. Along with this, I normalized the data to improve the efficency processing images. 

Here is an example of a traffic sign image before and after grayscaling/normalzing.

![alt text][image2]

My final model consisted of the following architecture layout and layers:

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
 
To train the model, I used the AdamOptimizer with a learning rate of .001, batch size of 64, and a total of 15 epochs. These were tested and tunned to get the best results. 
 
My final model results were:
* training set accuracy of 99%
* validation set accuracy of 96.1%
* test set accuracy of 95.1%

I decided to go with a well-known architecure model by using the LeNet model. I believed it would be good for several reasons. One of them being the LeNet architecture is known to be good for handling image processing. Another is the fact that we learned this method during our lessons so I was more familiar with it and understood more about what was happening behind the scenes of the model. I did apply the dropout method to the model, which slightly improved the results. 

### Test a Model on New Images

Here are six German traffic signs that I found on the web:

![alt text][image3] ![alt text][image4] ![alt text][image5] 
![alt text][image6] ![alt text][image7] ![alt text][image8]

In general, I picked these images due to them being on the more depective, clearer side of things. I expect my model should do a good job of classifying these specific images that I selected. 

The results of analyzing the new images was 100% for the 6 test images that I looked at. 

The guess success rate can be seen in the following image:

![alt text][image10]


As can be seen, the model was able to correctly identify the images that it was given. 




