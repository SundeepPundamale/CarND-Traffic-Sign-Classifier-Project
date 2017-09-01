# **Traffic Sign Recognition** 


# **Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/image1.jpeg "Traffic Sign 1"
[image5]: ./examples/image2.jpeg "Traffic Sign 2"
[image6]: ./examples/image3.jpeg "Traffic Sign 3"
[image7]: ./examples/image4.jpeg "Traffic Sign 4"
[image8]: ./examples/image5.jpg "Traffic Sign 5"
[image9]: ./examples/test_set.png "Test Set Distribution"
[image10]: ./examples/training_set.png "Training Set Distribution"
[image11]: ./examples/validation_set.png "Validation Set Distribution"
[image12]: ./examples/image_freq_label1.png "Unique Images Fruequency and Labels"
[image13]: ./examples/image_freq_label2.png "Unique Images Fruequency and Labels"
[image14]: ./examples/image_freq_label3.png "Unique Images Fruequency and Labels"
[image15]: ./examples/image_post_processing.png "Image after the preprocessing"
[image16]: ./examples/image_pre_processing.png "Image before the preprocessing"
[image17]: ./examples/augumented_training_set.png "Augumented training set"




## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup

#### The below writeup includes all the rubric points and each one is addressed. 

Here is a link to my [project code](https://github.com/SundeepPundamale/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. This section covers the basic summary of the data set. In the code, the analysis was done using python, numpy.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The dimension of the training set array is **4**
* The dimension of the validation set array is **4**
* The dimension of the test set array is **4**
* The size of the training set is **34799**
* The size of the validation set is **4410**
* The size of the test set is **12630**
* The shape of a traffic sign image is **(32, 32, 3)**
* The number of unique classes/labels in the data set is **43**

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. There are 3 bar charts, each one of them showing the distribution of images in Training, Test and Validation sets. In addition the mode of each sample set is  highlighted in the bar chart as a dotted red line. From this analysis it is evident that not all images have same frequency. There are few images which have a frequency as high as 2010 and few images as low as 180.

![alt text][image9]
![alt text][image10]
![alt text][image11]

In addition to plotting the distribution i also listed the unique images in the training set, which is 43 images. In addition i parsed the corresponding label names from the signnames.csv file. Finally i displayed all the 43 unique images with frequency and label in the title.

![alt text][image12]
![alt text][image13]
![alt text][image14]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I performed the following in my pre-processing steps: I first converted the images to grayscale. From the Convolutional Neural Network lesson i learned that the classifier performs better on grayscale image when color does not have significance. Therefore i first coverted the image to grayscale. In the next step i changed the shape of the image to (32,32,1) as the tensorflow placeholder had the same shape. I was getting en error when i trained the model without changing the shape. In the next step i normalised the image. I referred the following wiki page to understand the benefits of Normalisation: https://en.wikipedia.org/wiki/Normalization_(image_processing). I learned that normalisation helps to achieve consistency in dynamic range for a set of data.

Here is an example of a traffic sign image before and after the preprocessing.

![alt text][image16]
![alt text][image15]



At this point i trained the model and noticed that the validation accuracy was around ~0.93

It was evident from the Training set distribution that few images had a frequency as low as 180 and the mode of the distribution was 2010. So i appended the images to the unique classes such that all the images have a same frequency of 2010. Below is the image of the augumented training set distribution

![alt text][image17]

The difference between the original data set and the augmented data set is that every unique class has 2010 images. To augument the data i randomly duplicated the image, rotated the image or blurred the image and appended them to the existing data set


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					    | 
|:---------------------:|:-------------------------------------------------:| 
|Convolution layer 1   	| Input = 32x32x1, Output = 28x28x6   			    | 
|RELU Activation 1 		| 											 	    |
|Pooling layer 1		| Size 2x2 with Input = 28x28x6, Output = 14x14x6	|
|Convolution layer 2	| Input = 14x14x6, Output = 10x10x16 			    |
|RELU Activation 2	    | 		      									    |
|Pooling layer 2		| Size 2x2 with Input = 10x10x16,Output = 5x5x16    |									  |Flatten Layer 1		  |													  |	
|Dropout layer 1		| 	        									    |
|Fully Connected layer 1| Input = 400, Output = 120							|
|Droptout layer 2  		|												    |
|Fully Connected layer 2| Input = 120, Output = 84							|
|RELU Activation function 3|												|
|Fully Connected layer 3| Input = 84, Output = 43							|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used adam optimizer, a batch size of 128, 10 epochs and a learning rate of 0.001. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My starting point was the LeNet-5 implementation shown in the [classroom](https://github.com/udacity/CarND-LeNet-Lab/blob/master/LeNet-Lab-Solution.ipynb)

After changing the class size to 43 from the LeNet-5 implementation shown in the classroom i was able to achieve a validation accuracy of ~0.90. I then added 2 dropout layers as dropouts prevent overfitting. I was blocked prior to this stage and was unable to use the classroom LeNet-5 implementation and get an accuracy of ~0.89. I posted a query in the udacity forum and got pointers to change the unique class number in the default LeNet-5 implementation.  I then pre-processed the data by greyscaling and normalising and finally augumenting the data such that all the images have a same frequency and i got the following final results:

* Training set accuracy of 0.988
* Validation set accuracy of 0.960 
* Test set accuracy of 0.935


 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:


![alt text][image8] ![alt text][image4]
 
![alt text][image5] ![alt text][image6] 

![alt text][image7] 


After having an initial look at the random images i was under an impression that the priority sign could be difficult to classify because it has lots of unwanted information such as a wire running by, sky and the roof top of the house.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Priority Road      	| Priority Road   								| 
| Stop     				| Stop 										    |
| Turn Right Ahead		| Children crossing								|
| Right-of-way at the next intersection	      		| Right-of-way at the next intersection					 				|
| Road work			| Road work     							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 81st cell of the Ipython notebook.

The top five soft max probabilities were:

**Image1:**

| Prediction         	|     Probability	        					| 
|:---------------------:|:---------------------------------------------:| 
|  Priority road        |  0.99927								        | 
|  Yield   			    |  0.000575657									|
|  Roundabout mandatory	|  0.00014363       						    |
|  No passing	        |  6.85211e-06			                        |
|  Speed limit (50km/h)	|  1.18889e-06     							    |


**Image2:**

| Prediction         	|     Probability	        					| 
|:---------------------:|:---------------------------------------------:| 
|  Stop        			|  0.984738								        | 
|  Keep right   		|  0.00955064									|
|  No vehicles			|  0.00413769       						    |
|  Speed limit (60km/h)	|  0.000647092			                        |
|  Traffic signals	    |  0.000183445     							    |



**Image3:**

| Prediction         	|     Probability	        					| 
|:---------------------:|:---------------------------------------------:| 
|  Children crossing    |  0.893396								        | 
|  Speed limit (20km/h) |  0.0943099									|
|  Vehicles over 3.5 metric tons prohibited	|  0.00350181      			|
|  Ahead only			|  0.00302727			                        |
|  Speed limit (120km/h)|  0.00243131     							    |


**Image4:**

| Prediction         	|     Probability	        					| 
|:---------------------:|:---------------------------------------------:| 
|  Right-of-way at the next intersection|  0.999911						| 
|  Beware of ice/snow   |  6.34386e-05									|
|  Pedestrians			|  2.4617e-05       						    |
|  Road narrows on the right|  4.24202e-07		                        |
|  Children crossing	    |  2.64393e-07     							|


**Image5:**

| Prediction         	|     Probability	        					| 
|:---------------------:|:---------------------------------------------:| 
|  Road work            |  0.998457										| 
|  Go straight or left  |  0.00133303									|
|  Yield			    |  0.000168899       						    |
|  Beware of ice/snow   |  3.90485e-05		                      		|
|  Double curve	        |  6.61009e-07     							    |



