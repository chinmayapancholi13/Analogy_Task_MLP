
# coding: utf-8

import gzip
import os

import numpy as np
import scipy.spatial.distance as sp_dist
import random
import math
import tensorflow as tf

from sklearn.cross_validation import KFold

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.pairwise import euclidean_distances

import scipy.special as ss

analogyInputFile = "./word-analogy-dataset"
vectorTxtFile = "./glove.6B.300d.txt"
analogyTrainPath = "./wordRep/"
anaSoln = "./analogySolution.csv"

analogyDataset = [[stuff.strip() for stuff in item.strip().split('\n')] for item in open(analogyInputFile).read().split('\n\n')]

def multilayer_perceptron(x, weights, biases):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])     # Hidden layer with RELU activation
    layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])       # Hidden layer with RELU activation
    layer_2 = tf.nn.relu(layer_2)

    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']      # Output layer with linear activation
    return out_layer

#function to create mini_batches while training the MLP model
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]

    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)

    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)

        yield inputs[excerpt], targets[excerpt]


def analogyTask(inputDS=analogyDataset, outputFile = anaSoln): # add more arguments if required
    anaSoln_file_fp = open(outputFile, 'w')

    file_paths = []

    file_paths.append(analogyTrainPath + str("Pairs_from_Wikipedia_and_Dictionary/01-all-capital-cities.txt"))
    file_paths.append(analogyTrainPath + str("Pairs_from_Wikipedia_and_Dictionary/02-currency.txt"))
    file_paths.append(analogyTrainPath + str("Pairs_from_Wikipedia_and_Dictionary/03-city-in-state.txt"))
    file_paths.append(analogyTrainPath + str("Pairs_from_Wikipedia_and_Dictionary/04-man-woman.txt"))
    file_paths.append(analogyTrainPath + str("Pairs_from_Wikipedia_and_Dictionary/05-adjective-to-adverb.txt"))
    file_paths.append(analogyTrainPath + str("Pairs_from_Wikipedia_and_Dictionary/09-nationality-adjective.txt"))

    file_paths.append(analogyTrainPath + str("Pairs_from_WordNet/01-Antonym.txt"))
    file_paths.append(analogyTrainPath + str("Pairs_from_WordNet/02-MemberOf.txt"))
    file_paths.append(analogyTrainPath + str("Pairs_from_WordNet/03-MadeOf.txt"))
    file_paths.append(analogyTrainPath + str("Pairs_from_WordNet/04-IsA.txt"))
    file_paths.append(analogyTrainPath + str("Pairs_from_WordNet/05-SimilarTo.txt"))
    file_paths.append(analogyTrainPath + str("Pairs_from_WordNet/06-PartOf.txt"))
    file_paths.append(analogyTrainPath + str("Pairs_from_WordNet/07-InstanceOf.txt"))
    file_paths.append(analogyTrainPath + str("Pairs_from_WordNet/08-DerivedFrom.txt"))
    file_paths.append(analogyTrainPath + str("Pairs_from_WordNet/09-HasContext.txt"))
    file_paths.append(analogyTrainPath + str("Pairs_from_WordNet/10-RelatedTo.txt"))
    file_paths.append(analogyTrainPath + str("Pairs_from_WordNet/11-Attribute.txt"))
    file_paths.append(analogyTrainPath + str("Pairs_from_WordNet/12-Causes.txt"))
    file_paths.append(analogyTrainPath + str("Pairs_from_WordNet/13-Entails.txt"))

    all_examples = []
    file_index = 1

    train_vocab = set()     #contains all words in the folder 'word_rep'

    for curr_file_path in file_paths :
        for item in open(curr_file_path).read().splitlines() :
            elems = [item.split("\t")]

            train_vocab.add(elems[0][0])
            train_vocab.add(elems[0][1])

            elems.append(file_index)

            all_examples.append(elems)
        file_index = file_index + 1

    random.shuffle(all_examples)

    analogy_train_dict = {}     #contains vectors of words (for which vectors are present) in the folder word_rep

    vectorFile = open(vectorTxtFile , 'r')
    for line in vectorFile:
        if line.split()[0].strip() in train_vocab:
            analogy_train_dict[line.split()[0].strip()] = line.split()[1:]

    vectorFile.close()

    num_pos_examples = 1000 #50000
    num_neg_examples = 1000 #50000

    training_data_X = []
    training_data_Y = []

    pos_examples_done = 0
    neg_examples_done = 0

    for i in range(len(all_examples)):
        if pos_examples_done >= num_pos_examples and neg_examples_done >= num_neg_examples:
            break

        for j in range(i+1, len(all_examples)):
            if pos_examples_done >= num_pos_examples and neg_examples_done >= num_neg_examples:
                break

            word1 = all_examples[i][0][0]
            word2 = all_examples[i][0][1]
            word3 = all_examples[j][0][0]
            word4 = all_examples[j][0][1]

            if (word1 not in analogy_train_dict) or (word2 not in analogy_train_dict) or (word3 not in analogy_train_dict) or (word4 not in analogy_train_dict) :
                continue

            word1_vec = analogy_train_dict[word1]
            word2_vec = analogy_train_dict[word2]
            word3_vec = analogy_train_dict[word3]
            word4_vec = analogy_train_dict[word4]

            word1_vec = np.array(word1_vec, dtype=float)
            word2_vec = np.array(word2_vec, dtype=float)
            word3_vec = np.array(word3_vec, dtype=float)
            word4_vec = np.array(word4_vec, dtype=float)

            net_input_vec = np.subtract(word1_vec, word2_vec)
            net_input_vec_extended = np.append(net_input_vec, np.subtract(word3_vec, word4_vec))

            if pos_examples_done < num_pos_examples and all_examples[i][1] == all_examples[j][1] :
                training_data_X.append(net_input_vec_extended)
                training_data_Y.append(1)
                pos_examples_done = pos_examples_done + 1

            if neg_examples_done < num_neg_examples and all_examples[i][1] != all_examples[j][1] :
                training_data_X.append(net_input_vec_extended)
                training_data_Y.append(0)
                neg_examples_done = neg_examples_done + 1

    training_data_X = np.array(training_data_X)
    training_data_Y = np.array(training_data_Y)

    # Parameters
    learning_rate = 0.001
    training_epochs = 2 #5
    batch_size = 100
    display_step = 1

    # Network Parameters
    n_hidden_1 = 300 # 1st layer number of features
    n_hidden_2 = 300 # 2nd layer number of features
    n_input = 600
    n_classes = 1

    training_data_X = np.array(training_data_X).reshape(len(training_data_Y) , n_input)
    training_data_Y = np.array(training_data_Y).reshape(len(training_data_Y) , n_classes)

    out_prediction = np.empty((0, 1))
    k_fold_valid_num = 0

    for curr_f_num in k_fold_valid_function(training_data_X, training_data_Y) :
        X_train, X_test, y_train, y_test = curr_f_num
        # tf Graph input
        x = tf.placeholder("float", [None, n_input])
        y = tf.placeholder("float", [None, n_classes])
        # Store layers weight & bias
        weights = {
            'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])*np.sqrt(2./(n_input+n_hidden_1))),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])*np.sqrt(2./(n_hidden_1+n_hidden_2))),
            'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes])*np.sqrt(2./(n_hidden_2+n_classes)))
        }
        biases = {
            'b1': tf.Variable(tf.zeros([n_hidden_1])),
            'b2': tf.Variable(tf.zeros([n_hidden_2])),
            'out': tf.Variable(tf.zeros([n_classes]))
        }
        # Construct model
        pred = multilayer_perceptron(x, weights, biases)
        # Define loss and optimizer
        cost = tf.reduce_mean(tf.contrib.losses.hinge_loss(logits=pred, labels=y))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
        # Initializing the variables
        init = tf.global_variables_initializer()

        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)

            # Training cycle
            for epoch in range(training_epochs):
                avg_cost = 0.
                # Loop over all batches
                for batch in iterate_minibatches(X_train, y_train, batch_size):
                # for i in range(total_batch):
                    batch_x, batch_y = batch
                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                    # Compute average loss
                    avg_cost += c / batch_size
                # Display logs per epoch step
                if epoch % display_step == 0:
                    print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

            k_fold_valid_num = k_fold_valid_num + 1
            print("Fold Iteration : ",k_fold_valid_num," : Optimization Finished")

            # Testing the model
            test_data_x = tf.placeholder("float", [None, n_input])
            test_data_pred = multilayer_perceptron(test_data_x, weights, biases)

            test_data_pred = sess.run(test_data_pred, feed_dict = {test_data_x:X_test})
            out_prediction = np.vstack((out_prediction, test_data_pred))

    correct_prediction = 0.0
    for i  in range(np.shape(out_prediction)[0]):
        exp_val = ss.expit(out_prediction[i])
        if (exp_val >=0.5 and training_data_Y[i]==1) or (exp_val <0.5 and training_data_Y[i]==0) :
            correct_prediction = correct_prediction + 1

    ana_task_accuracy = float(correct_prediction) / float(np.shape(out_prediction)[0])
    print ("ana_task_accuracy", ana_task_accuracy)

    training_epochs = 2 #20

    # tf Graph input
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])*2/(np.sqrt(n_input + n_hidden_1))),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])*2/(np.sqrt(n_hidden_1 + n_hidden_2))),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes])*2/(np.sqrt(n_hidden_2 + n_classes)))
    }
    biases = {
        'b1': tf.Variable(tf.zeros([n_hidden_1])),
        'b2': tf.Variable(tf.zeros([n_hidden_2])),
        'out': tf.Variable(tf.zeros([n_classes]))
    }

    # Construct model
    pred = multilayer_perceptron(x, weights, biases)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.contrib.losses.hinge_loss(logits=pred, labels=y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initializing the variables
    init = tf.global_variables_initializer()

    # print(training_data_Y.shape)
    total_number_of_batches = int(training_data_Y.shape[0] / batch_size)
    print(total_number_of_batches)

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = float(0.)

            for batch in iterate_minibatches (training_data_X, training_data_Y, batch_size, shuffle=True):
                batch_x, batch_y = batch
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                # Compute average loss
                avg_cost = avg_cost + float(c) / float(total_number_of_batches)

            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

        print("Optimization Finished!")

        total_test_labelled = 0.0
        total_test_correct = 0.0

        analogy_test_dict = {}

        analogyList = [thing.strip()for item in inputDS for stuff in item[0:6] for thing in stuff.split()]

        vectorFile = open(vectorTxtFile , 'r')
        for line in vectorFile:
            if line.split()[0].strip() in analogyList:
                analogy_test_dict[line.split()[0].strip()] = line.split()[1:]

        vectorFile.close()

        for item in analogyDataset:
            word1 = item[0].split(" ")[0].strip()
            word2 = item[0].split(" ")[1].strip()

            word3 = item[1].split(" ")[0].strip()
            word4 = item[1].split(" ")[1].strip()

            word5 = item[2].split(" ")[0].strip()
            word6 = item[2].split(" ")[1].strip()

            word7 = item[3].split(" ")[0].strip()
            word8 = item[3].split(" ")[1].strip()

            word9 = item[4].split(" ")[0].strip()
            word10 = item[4].split(" ")[1].strip()

            word11 = item[5].split(" ")[0].strip()
            word12 = item[5].split(" ")[1].strip()

            in_vocab =True

            if (word1 not in analogy_test_dict) or (word2 not in analogy_test_dict) or (word3 not in analogy_test_dict) or (word4 not in analogy_test_dict) or (word5 not in analogy_test_dict) or (word6 not in analogy_test_dict) or (word7 not in analogy_test_dict) or (word8 not in analogy_test_dict) or (word9 not in analogy_test_dict) or (word10 not in analogy_test_dict) or (word11 not in analogy_test_dict) or (word12 not in analogy_test_dict) :
                continue

            total_test_labelled = total_test_labelled + 1

            if item[6]=='a':
                predicted_value = 1
            elif item[6]=='b':
                predicted_value = 2
            elif item[6]=='c':
                predicted_value = 3
            elif item[6]=='d':
                predicted_value = 4
            elif item[6]=='e':
                predicted_value = 5

            word1_vec = analogy_test_dict[word1]
            word2_vec = analogy_test_dict[word2]
            word1_vec = np.array(word1_vec, dtype=float)
            word2_vec = np.array(word2_vec, dtype=float)
            net_input_vec = np.subtract(word1_vec, word2_vec)

            #option1
            word3_vec = analogy_test_dict[word3]
            word4_vec = analogy_test_dict[word4]
            word3_vec = np.array(word3_vec, dtype=float)
            word4_vec = np.array(word4_vec, dtype=float)
            net_input_vec_extended1 = np.append(net_input_vec, np.subtract(word3_vec, word4_vec))
            net_input_vec_extended1 = net_input_vec_extended1.reshape(1, 600)

            #option2
            word5_vec = analogy_test_dict[word5]
            word6_vec = analogy_test_dict[word6]
            word5_vec = np.array(word5_vec, dtype=float)
            word6_vec = np.array(word6_vec, dtype=float)
            net_input_vec_extended2 = np.append(net_input_vec, np.subtract(word5_vec, word6_vec))
            net_input_vec_extended2 = net_input_vec_extended2.reshape(1, 600)

            #option3
            word7_vec = analogy_test_dict[word7]
            word8_vec = analogy_test_dict[word8]
            word7_vec = np.array(word7_vec, dtype=float)
            word8_vec = np.array(word8_vec, dtype=float)
            net_input_vec_extended3 = np.append(net_input_vec, np.subtract(word7_vec, word8_vec))
            net_input_vec_extended3 = net_input_vec_extended3.reshape(1, 600)

            #option4
            word9_vec = analogy_test_dict[word9]
            word10_vec = analogy_test_dict[word10]
            word9_vec = np.array(word9_vec, dtype=float)
            word10_vec = np.array(word10_vec, dtype=float)
            net_input_vec_extended4 = np.append(net_input_vec, np.subtract(word9_vec, word10_vec))
            net_input_vec_extended4 = net_input_vec_extended4.reshape(1, 600)

            #option5
            word11_vec = analogy_test_dict[word11]
            word12_vec = analogy_test_dict[word12]
            word11_vec = np.array(word11_vec, dtype=float)
            word12_vec = np.array(word12_vec, dtype=float)
            net_input_vec_extended5 = np.append(net_input_vec, np.subtract(word11_vec, word12_vec))
            net_input_vec_extended5 = net_input_vec_extended5.reshape(1, 600)

            predicted_val1 = multilayer_perceptron(x, weights, biases)
            predicted_val2 = multilayer_perceptron(x, weights, biases)
            predicted_val3 = multilayer_perceptron(x, weights, biases)
            predicted_val4 = multilayer_perceptron(x, weights, biases)
            predicted_val5 = multilayer_perceptron(x, weights, biases)

            predicted_val1 = sess.run(predicted_val1, feed_dict={x:net_input_vec_extended1})
            predicted_val2 = sess.run(predicted_val2, feed_dict={x:net_input_vec_extended2})
            predicted_val3 = sess.run(predicted_val3, feed_dict={x:net_input_vec_extended3})
            predicted_val4 = sess.run(predicted_val4, feed_dict={x:net_input_vec_extended4})
            predicted_val5 = sess.run(predicted_val5, feed_dict={x:net_input_vec_extended5})

            val_array = []
            val_array.append(predicted_val1[0][0])
            val_array.append(predicted_val2[0][0])
            val_array.append(predicted_val3[0][0])
            val_array.append(predicted_val4[0][0])
            val_array.append(predicted_val5[0][0])

            print val_array
            print np.argmax(val_array)
            print predicted_value

            option_pred_int =  np.argmax(val_array) + 1
            option_pred_char = 'a'
            if option_pred_int == 1 :
                option_pred_char = 'a'
            elif option_pred_int == 2 :
                option_pred_char = 'b'
            elif option_pred_int == 3 :
                option_pred_char = 'c'
            elif option_pred_int == 4 :
                option_pred_char = 'd'
            elif option_pred_int == 5 :
                option_pred_char = 'e'

            option_actual = 'a'
            if predicted_value == 1 :
                option_actual = 'a'
            elif predicted_value == 2 :
                option_actual = 'b'
            elif predicted_value == 3 :
                option_actual = 'c'
            elif predicted_value == 4 :
                option_actual = 'd'
            elif predicted_value == 5 :
                option_actual = 'e'

            str_analogy = str(word1)+str(" ") + str(word2) + str(",") + str(option_actual) + str(",") + str(option_pred_char) + "\n"
            anaSoln_file_fp.write(str_analogy)

            if np.argmax(val_array) == predicted_value - 1:
                total_test_correct = total_test_correct + 1

        test_accuracy = float(total_test_correct) / float(total_test_labelled)

        print total_test_correct
        print total_test_labelled
        print test_accuracy

    anaSoln_file_fp.close()
    return ana_task_accuracy    #return the accuracy of the model after 5 fold cross validation
    """
    Output a file, analogySolution.csv with the following entries
    Query word pair, Correct option, predicted option
    """

#function to perform k_fold_validation (here k=5)
def k_fold_valid_function(X,Y):
    scores = []
    kf = KFold(n = len(X), n_folds=5)
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        yield X_train, X_test, Y_train, Y_test

def main():
    anaSim = analogyTask()

if __name__ == '__main__':
    main()
