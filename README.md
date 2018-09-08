# SVM ALGORITHM

## preprocess.py:
Input: raw data

Output: document term matrix

Overview: Contains functions that takes the raw data and produces document-term matrix

## SVM.py:
Input: document-term matrix

Output: trained model and predictions with model

Overview: Contains an svm class use to build, train and predict a given data set. It also has a function
            for creating the confusion matrix

## Packages:
The following packages are required:

numpy for scientific computing

pandas for loading files

scipy for mathematics, science and engineering calculations

nltk for natural language processing

scikit-learn for machine learning algorithms

# CITATIONS:
I consulted a matlab code from Machine Learning course on coursera taught by Stanford University professor
Andrew Ng.

## Reference
* [Sequential Minimal Optimization- A Fast Algorithm for Training Support Vector Machines](https://www.microsoft.com/en-us/research/publication/sequential-minimal-optimization-a-fast-algorithm-for-training-support-vector-machines/)

Improvements to platt's SMO algorithm for SVM classifier design

CS 229, Autumn 2009 | The Simplified SMO Algorithm
