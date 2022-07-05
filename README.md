# classifiers
Naive Bayes Classifier and k-nearest neighbour classifier

These algorithms both work on a given data set of .png images. They follow the basic methods of data classifiaction; at first they learn the given data set (or a portion of it) and then they can be used.

-----------------------------------------------------------------------------------------

The file classifier.py is meant to be called from the command line with these arguments:
>> python3.8 classifier.py -h
usage: classifier.py [-h] (-k K | -b) [-o filepath] train_path test_path

Learn and classify image data.

positional arguments:

train_path   path to the training data directory

test_path    path to the testing data directory

optional arguments:
- -h, --help   show this help message and exit
- -k K         run k-NN classifier (if k is 0 the code may decide about proper K by itself
- -b           run Naive Bayes classifier
- -o filepath  path (including the filename) of the output .dsv file with the results
  
----------------------------------------------------------------------------------------

NOTE: right now, if the progeramme is to be run with the data from the uploaded .zip file, it must be done in this way:
  
Bayes: 
- 2 folders must be created: train_path and test_path
- put all the data to train_path and NONE to test_path
  
KNN:
- 2 folders must be created: train_path and test_path
- split the data between train_path and test_path
