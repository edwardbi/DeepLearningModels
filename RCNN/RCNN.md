# RCNN
The Tensorflow with tflearn implementation of the RCNN model

**The Enviroment:**
The file is working with Tensorflow version 0.7.1. Python version 2.7 and scikit-learn dev distribution package. Also need to install the tflearn project from github at https://github.com/tflearn/tflearn. Notice that the tflearn requires Tensorflow version at least 0.7.0. For better compatiblity, you may want to install Tensorflow 0.8.0 or above. Finally, you need the 
selective search project from python. You may get it by pip install selectivesearch. 

**Training Input:**
The work trains with the 17 flowers dataset. The data can be obtained here: https://github.com/ck196/tensorflow-alexnet. Download the project and use the 17flowers.tar.gz file's data as the training data. 

**The Code:**
There are four files in the code. The train_alexnet.py uses the 17flowers image folder with the train_list.txt file to perform the pre-training of the Alexnet. This is the file that need to be run first. After it generates the model_save.model file, you can run the fine_tune_RCNN.py file, which fine-tunes the model with the 2flowers dataset as well as the two txt files in
the svm_train folder. Up on its completion, a file named fine_tune_model_save.model is created as the fine-tuned model result. Finally, you may run the RCNN_output.py file to classify images. In the main part of the file, change the image name to test on different images. Now, I fine-tuned the image with only two categories of flowers, namely, the pansy and the tulip. 
