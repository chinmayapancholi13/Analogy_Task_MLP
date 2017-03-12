# Analogy_Task_MLP
Given a pair of words a : b we find out the pair of words among five given pair of words, which one is more appropriate as far as analogy is concerned. The glove vectors used in the task have been downloaded from http://nlp.stanford.edu/projects/glove/ (we have used the file with 6 billion tokens).

We learn a deep learning model for the task (we have used a 2 hidden layer MLP here). Finally, we report the accuracy of the model after performing 5-fold cross validation. 

For training, we use the training files in the folder **./wordRep**. In the files present in this folder, all the pairs belonging to same category have been given in a single file. For validating the model we use the file **Word-analogy-dataset**.

