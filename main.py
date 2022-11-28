import numpy as np


def create_feature_matrix(vectors, vocabulary):
    matrix = []
    for vector in vectors:
        feature_vector  = []
        s = vector.split(" ")
        for word in vocabulary:
            if word in s:
                feature_vector.append(1)
                continue
            feature_vector.append(0)
        matrix.append(feature_vector)

    return matrix

def test_perceptron(test_vector_table, train_weights, test_labels, test_accuracy, test_mistakes):
    mistakes = 0
    for index in range(len(test_vector_table)):
        y = np.dot(test_vector_table[index], train_weights)
        if(y > 0):
            y = 1
        else:
            y = 0
        if(y != test_labels[index]):
                mistakes += 1

    test_accuracy.append(1 - mistakes/len(test_vector_table))
    test_mistakes.append(mistakes)

def test_avg_perceptron(test_vector_table, avg_weights, test_labels):
    mistakes = 0
    for index in range(len(test_vector_table)):
        y = np.dot(test_vector_table[index], avg_weights)
        if(y > 0):
            y = 1
        else:
            y = 0
        if(y != test_labels[index]):
                mistakes += 1

    avg_test_accuracy = (1 - mistakes/len(test_vector_table))
    return avg_test_accuracy, mistakes

def multiClassPerceptron(data_vectors, mistakes, weights, labels, test_data_vector, test_labels):
    test_mistakes = []
    for iteration in range(20):
        for vector in range(len(data_vectors)):
            argmax = -1000000000
            prediction = -1
            for i in range(26):
                yhat = np.dot(data_vectors[vector], weights[i])
                if(yhat > argmax):
                    argmax =  yhat
                    prediction = i
            
            if labels[vector] == prediction:
                continue

            mistakes[iteration] += 1
            weights[prediction] = np.subtract(weights[prediction],data_vectors[vector])
            weights[labels[vector]] = np.add(weights[labels[vector]],data_vectors[vector])
        test_mistakes.append(test_MultiClassPerceptron(test_data_vector, weights, test_labels))

    return mistakes,test_mistakes

def test_MultiClassPerceptron(test_vector_table, weights, test_labels):
    mistakes = 0
    for index in range(len(test_vector_table)):
        argmax = -10000000
        prediction = -1
        for i in range(26):
            yhat = np.dot(test_vector_table[index], weights[i])
            if(yhat > argmax):
                argmax = yhat
                prediction = i
        
        if(prediction != test_labels[index]):
            mistakes += 1

    return mistakes   

output_file = open("output.txt", "w+")

train_file = open("data/fortune-cookie-data/traindata.txt", "r")
train_label_file = open("data/fortune-cookie-data/trainlabels.txt", "r")
test_file = open("data/fortune-cookie-data/testdata.txt", "r")
test_label_file = open("data/fortune-cookie-data/testlabels.txt", "r")
stop_file = open("data/fortune-cookie-data/stoplist.txt", "r")

train_data = train_file.read().replace("\n", " ").split(" ")
train_labels = list(train_label_file.read().split("\n"))
test_data = test_file.read()
test_labels = list(test_label_file.read().split("\n"))
stop_words = stop_file.read()

train_labels = [int(label) for label in train_labels]
test_labels = [int(label) for label in test_labels]

  
stop_word_list = stop_words.split("\n")


train_vocabulary = list(set(train_data)-(set(stop_word_list)))
train_vocabulary.sort()

if(train_vocabulary[0] == ""):
    train_vocabulary = train_vocabulary[1:]

train_file.seek(0)
test_file.seek(0)
train_data_vectors = train_file.read().splitlines()
test_data_vectors = test_file.read().splitlines()

train_vector_table = create_feature_matrix(train_data_vectors, train_vocabulary)

test_vector_table = create_feature_matrix(test_data_vectors, train_vocabulary)

vector_len = len(train_vocabulary)
train_weights = [0]*vector_len
lr = 1
mistakes = [0]*20
test_mistakes = []
test_accuracy = []
avg_weights = train_weights
for iteration in range(20):
    for index in range(len(train_vector_table)):
        y = np.dot(train_vector_table[index], train_weights)
        # print(train_weights)
        if(y > 0):
            y = 1
        else:
            y = 0
        if(y != train_labels[index]):
            mistakes[iteration] += 1
            train_weights = np.add(train_weights, np.dot(lr*(train_labels[index] - y),train_vector_table[index]))
    test_perceptron(test_vector_table, train_weights, test_labels, test_accuracy, test_mistakes)
    avg_weights += train_weights 

train_accuracy = [1 - (mistakes[i]/len(train_vector_table)) for i in range(len(mistakes))]



lines = ["Iteration-" + str(i+1) + " " + str(mistakes[i]) + "\n" for i in range(len(mistakes))]
output_file.writelines(lines)
lines = ["iteration-" + str(i+1) + " " + str(train_accuracy[i]) + " " + str(test_accuracy[i]) + "\n" for i in range(20)]
output_file.writelines(lines)

avg_test_accuracy, avg_test_mistakes = test_avg_perceptron(test_vector_table, avg_weights, test_labels)

lines = [str(train_accuracy[-1]) + " " + str(avg_test_accuracy)]
output_file.writelines(lines)

# printing the data
train_file.close()
stop_file.close()

#part 1 done

#part 2
train_file = open("data/OCR-data/ocr_train.txt", "r")
test_file = open("data/OCR-data/ocr_test.txt", "r")

lr = 1

train_data = train_file.read().split()
count  = 1
train_label_array = []
processed_train_data = []
for data in train_data:
    if count == 2:
        processed_train_data.append(data[2:])
        count += 1
        continue
    if count == 3:
        train_label_array.append(data)
        count += 1
        continue
    if count == 4:
        count = 1
        continue
    count += 1

train_data_vectors = []
for data in processed_train_data:
    train_data_vectors.append([int(bin) for bin in data])

train_label_array = [ord(label.lower()) - 97 for label in train_label_array]
weights = [[0]*len(train_data_vectors[0])]*26
mistakes = [0]*20

test_data = test_file.read().split()
count  = 1
test_label_array = []
processed_test_data = []
for data in test_data:
    if count == 2:
        processed_test_data.append(data[2:])
        count += 1
        continue
    if count == 3:
        test_label_array.append(data)
        count += 1
        continue
    if count == 4:
        count = 1
        continue
    count += 1

test_data_vectors = []
for data in processed_test_data:
    test_data_vectors.append([int(bin) for bin in data])

test_label_array = [ord(label.lower()) - 97 for label in test_label_array]
mistakes, test_mistakes = multiClassPerceptron(train_data_vectors,mistakes, weights, train_label_array, test_data_vectors, test_label_array)

lines = ["Iteration-" + str(i+1) + " " + str(mistakes[i]) + "\n" for i in range(len(mistakes))]
output_file.writelines(lines)
lines = ["iteration-" + str(i+1) + " " + str(1-mistakes[i]/len(train_data_vectors)) + " " + str(1-test_mistakes[i]/len(test_data_vectors)) + "\n" for i in range(20)]
lines[-1] = lines[-1][:-1]
output_file.writelines(lines)
