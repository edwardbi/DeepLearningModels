# DM
The Tensorflow implementation of the DM model from Word2Vec

**The Enviroment:**
The file is working with Tensorflow version 0.7.1. Python version 2.7 and scikit-learn dev distribution package is needed as it has the TSNE algorithm implemented. To run the code, you may want to install ipython notebook to your machine as well. Also, for the Gensim file, you need Gensim library installed

**Training Input:**
The work assumes the wikipedia corpus for training. The corpus can be download from https://blog.lateral.io/2015/06/the-unknown-perils-of-mining-wikipedia/

**The Code:**
The code is modified from the CBOW model previously made using Tensorflow. The base code can be find under the CBOW folder . The code uses only the DM model with negative sampling techniques as training method. For testing the accuracy of the model, I randomly choose a sentence and find its similar ones to see if it make sense. 
For the Gensim code, it is used straight as a black box tool from Gensim. 
