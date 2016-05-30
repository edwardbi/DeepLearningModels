# CBOW
The Tensorflow implementation of the CBOW model from Word2Vec

**The Enviroment:**
The file is working with Tensorflow version 0.7.1. Python version 2.7 and scikit-learn dev distribution package is needed as it has the TSNE algorithm implemented. The link to the scikit-learn dev can be find here: https://github.com/scikit-learn/scikit-learn.git . To run the code, you may want to install ipython notebook to your machine as well.

**Training Input:**
The work assumes the text8.zip corpus for training. The corpus can be download from http://mattmahoney.net/dc/text8.zip

**The Code:**
The code is modified from the Skip-Gram model provided by Tensorflow. The code is based off the word2vec_basic.py from here: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/word2vec/word2vec_basic.py . The code uses onlly the CBOW model with negative sampling techniques as training method. For testing the accuracy of the model, aside from the vector algebra method provided with the code, one can also use the TSNE plot. To use the TSNE plot, you need to have the dev version of the scikit-learn, and have the project inside the scikit-learn folder for tsne to be able to show. Before plotting the tsne, change the tsne_plot flag from False to True. 
