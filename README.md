# Butterflies

Project Description:

This project involves creating an image classification model for categorizing different species of butterflies by testing out different libraries and methods.

Alara Balasaygun (Applied Math) - Having never worked with either image classification or neural networks, I am interested in the new scope and skills working on this project will provide in both areas. Gaining more experience with deep learning and putting into practice the libraries of tensorflow and pytorch is also important to me as I hope to further gain deep learning knowledge. In my Math 151a class that I’m currently taking, my TA has discussed his career path as a math graduate where he works heavily with graphics. While image classification isn’t quite that, I do find the relationship between math and imaging quite interesting and hope to further explore it here. As a nat geo lover, butterfly species classification is a fun task to focus on while putting the above into practice.

Miguel (Cognitive Science) - In 16A I really enjoyed working with machine learning, as well as our comprehensive project at the end of the quarter as I felt that it deepened my understanding of the subject. I hope to continue to develop my understanding through image classification and other machine learning techniques. I know that image classification is an important topic currently, and I hope that the skills I learn through these projects will be helpful in my future endeavors. 

Amay Jain (Applied Math) - I am interested in this project because I have only worked with tabular data before and I want to learn how to do image classification, which has a slightly different approach from classification involving numerical/categorical predictors. It was interesting to learn both in this class and my math classes on how we could compress images with SVD techniques and we aim for a similar approach in this project.

Resources:

We will first try our models in Jupyter, and if the computing power is not enough then we will switch to Google Colab. Since we are working with images we will mainly be using the GPU resources provided, and ideally compress the images in the data processing stage. For final purposes, we will convert everything to .py scripts using VS Code. We will be using this butterfly dataset taken from kaggle, which holds 2786 unique images and is 43.48 kB large. The dataset is well maintained, being updated annually and having its last update 10 months ago. 


Previous Work: 

The butterfly datasets’ page on Kaggle has several user-created projects employing image classification. They vary in complexity, with 57 total projects uploaded. The projects feature a variety of machine learning and neural network techniques, with some employing Pytorch, Transfer Learning, and MobileNetV2. Since we will be working with Pytorch, these past projects will be relevant as we attempt to design our attempt at image classification. These projects are pretty comprehensive in scope, utilizing visualization techniques, confusion matrix, and multiple machine learning and neural network techniques so they should provide a useful guideline as we complete our project.

Tools and skills:

We will use either Tensorflow or Pytorch as our deep learning library for building the neural network. Techniques from class like gradient descent and principal component analysis will aid in adjusting the weights of features and slimming our network down to its most essential features. When it comes to the image classification side, we will likely use the OpenCV and/or Scikit-Image libraries to work with the image data. In particular, to shrink the pixels of the images and open them, we intend to utilize the Pillow library. In terms of evaluating the effectiveness of our neural network, we will look at calculating the loss as well as cross validation techniques.

Learn:

We will learn how to do data pre-processing with images (compression, resizing of pixels), how to use convolutions that scan the images to extract important features, potential loss functions and gradient descent algorithms specialized for images, and how to compare different models/libraries with various metrics.










Timeline:

All of us will schedule time each week to work on the project.

Week 5 - Learn how to open up image data and resize the pictures

Week 6 - Start by implementing 1 or 2 models

Week 7 - Apply loss functions and other hyperparameter algorithms to clean up models

Week 8 - Use other models to compare to base models and compare metrics (accuracy, loss, etc.)

Week 9 - Obtain feedback from professor and TA, Start making presentation

Week 10 - Finalize project on GitHub and wrap presentation 





