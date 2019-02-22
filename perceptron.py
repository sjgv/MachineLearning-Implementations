'''
Written By Salvador Gutierrez
'''
import copy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#df = pd.read_csv('diabetes.train', delim_whitespace=True, header=None)
#dobj = Data(fpath = 'diabetes.train')

'''
Majority Baseline
'''
def majorityOracle(truth, data):
    counter = [0,0]
    for row in data:
        label = int(row[0])
        prediction = truth
        if label == prediction:
            counter[0]+=1
        else:
            counter[1]+=1
    return float(counter[0])/float((counter[0]+counter[1]))

'''
This fn takes in a W and a dataset 
makes predictions on dataset using W
reports accuracy
'''
def oracle(W, data, bias):
    counter = [0,0]
    for row in data:
        label = int(row[0])
        prediction = sgnOf(wDotX(W,row,1,bias)[0])
        if label == prediction:
            counter[0]+=1
        else:
            counter[1]+=1
    return float(counter[0])/float((counter[0]+counter[1]))
'''
    print("Correct: ", counter[0])
    print("Incorrect:", counter[1])
    print("Testing Accuracy:", float(counter[0])/float((counter[0]+counter[1])))
'''
    

'''
Takes: float
returns: int
'''
def sgnOf(dotP):
    if np.sign(dotP) == -1:
        return -1
    else:
        return 1

'''
Takes in a dic W of weight:value and a row List of strings
Returns: [float, {}, {}] 
second val is negW of first original x vector
third val is scaledW
'''
def wDotX(W,row,r, bias):
    negW = {}
    scaledX = {}
    dotP = 0.0000001
    #Start at 1 because 0 is truth
    for i in range(1,len(row)):
        elm = row[i]
        elm_idx = elm.index(':')
        label = int(row[0])
        #An element has format 'idx:value'
        val_idx = int(elm[:elm_idx])
        val = float(elm[elm_idx + 1:len(elm)])
        #Compute negative 
        if label < 0:
            negW[val_idx] = (val*-1*r)                
        #Compute scaled vector
        scaledX[val_idx] = (val * r)
        #Compute dot product
        if val_idx in W:
           dotP += (W[val_idx] * val)
        #If it's not in W the value is zero
        else:
            continue
    #try some bias
    dotP += bias
    return [dotP, negW, scaledX]    
    
def perceptronAvgHW(data, r, bias, W_prime, A_prime, bias_avg):
    W = W_prime #Our weight vector is zero at first
    A = A_prime
    #For Debugging *******************
    timesUpdated = 0.0
    totalRows = float(len(data))
    #****************************
    for row in data:
        label = int(row[0])
        #Update learning rate
        #Make a prediction based on the sign of (w_t(x_i))
        dotP_negW = wDotX(W, row, r,bias)
        if((dotP_negW[0] * label) < 0 ):
            timesUpdated += 1
            #Update bias
            bias += (r*label)
            #Update W vector 
            if label < 0:
                #We calculated the negW while doing the dot product
                #Add to W 
                for key,value in dotP_negW[1].items():
                    if key in W:
                        W[key] += value
                    else:
                        W[key] = value
            #Otherwise it's just the scaled version
            else:
                for key,value in dotP_negW[2].items():
                    if key in W:
                        W[key] += value
                    else:
                        W[key] = value
        #No matter what update A
        for key,value in W.items():
            if key in A:
                A[key] += value
            else:
                A[key] = value

        #No matter what update bias_avg
        bias_avg += bias
    print(timesUpdated)
    return [W,A, bias, bias_avg]


'''
MARGIN parameter
'''
def perceptronMarginHW(data, r, bias, W_prime, mu):
    W = W_prime #Our weight vector is zero at first
    #For Debugging *******************
    timesUpdated = 0.0
    totalRows = float(len(data))
    #****************************
    for row in data:
        label = int(row[0])
        #Make a prediction based on the sign of (w_t(x_i))
        dotP_negW = wDotX(W, row, r,bias)
        if((dotP_negW[0] * label) < mu):
            timesUpdated += 1
            #Update bias
            bias += (r*label)
            #Update W vector 
            if label < 0:
                #We calculated the negW while doing the dot product
                #Add to W 
                for key,value in dotP_negW[1].items():
                    if key in W:
                        W[key] += value
                    else:
                        W[key] = value
            #Otherwise it's just the scaled version
            else:
                for key,value in dotP_negW[2].items():
                    if key in W:
                        W[key] += value
                    else:
                        W[key] = value
    #print("Training Accuracy:", (totalRows-timesUpdated)/totalRows)
    print(timesUpdated)
    return [W, bias]


'''
Most robust implementation
'''
def perceptronDecayingHW(data, r, bias, W_prime, epochs):
    W = W_prime #Our weight vector is zero at first
    #For Debugging *******************
    timesUpdated = 0.0
    totalRows = float(len(data))
    etta = float(r)/(1+epochs)
    #****************************
    #Decaying etta/1+t learning rate
    for row in data:
        label = int(row[0])
        #Make a prediction based on the sign of (w_t(x_i))
        dotP_negW = wDotX(W, row, etta,bias)
        if((dotP_negW[0] * label) < 0 ):
            timesUpdated += 1
            #Update bias
            bias += (etta*label)
            #Update W vector 
            if label < 0:
                #We calculated the negW while doing the dot product
                #Add to W 
                for key,value in dotP_negW[1].items():
                    if key in W:
                        W[key] += value
                    else:
                        W[key] = value
            #Otherwise it's just the scaled version
            else:
                for key,value in dotP_negW[2].items():
                    if key in W:
                        W[key] += value
                    else:
                        W[key] = value
        epochs += 1
    #print("Training Accuracy:", (totalRows-timesUpdated)/totalRows)
    print(timesUpdated)
    return [W, bias, epochs]

    
def perceptronHW(data, r, bias, W_prime):
    W = W_prime #Our weight vector is zero at first
    #For Debugging *******************
    timesUpdated = 0.0
    totalRows = float(len(data))
    #****************************
    for row in data:
        label = int(row[0])
        #Update learning rate
        #Make a prediction based on the sign of (w_t(x_i))
        dotP_negW = wDotX(W, row, r,bias)
        if((dotP_negW[0] * label) < 0 ):
            timesUpdated += 1
            #Update bias
            bias += (r*label)
            #Update W vector 
            if label < 0:
                #We calculated the negW while doing the dot product
                #Add to W 
                for key,value in dotP_negW[1].items():
                    if key in W:
                        W[key] += value
                    else:
                        W[key] = value
            #Otherwise it's just the scaled version
            else:
                for key,value in dotP_negW[2].items():
                    if key in W:
                        W[key] += value
                    else:
                        W[key] = value
    print("Times updated:", timesUpdated)
    print("Training Accuracy:",(totalRows-timesUpdated)/totalRows)
    return [W, bias]
    

'''
Most robust implementation
'''
def perceptronDecaying(data, r, bias):
    W = {} #Our weight vector is zero at first
    #For Debugging *******************
    timesUpdated = 0.0
    totalRows = float(len(data))
    #****************************
    #Decaying etta/1+t learning rate
    epochs = 0
    for row in data:
        label = int(row[0])
        #Update learning rate
        etta = float(r)/(1+epochs)
        #Make a prediction based on the sign of (w_t(x_i))
        dotP_negW = wDotX(W, row, etta,bias)
        prediction = sgnOf(dotP_negW[0])
        if prediction != label:
            timesUpdated += 1
            #Update bias
            bias += (etta*label)
            #Update W vector 
            if label < 0:
                #We calculated the negW while doing the dot product
                #Add to W 
                for key,value in dotP_negW[1].items():
                    if key in W:
                        W[key] += value
                    else:
                        W[key] = value
            #Otherwise it's just the scaled version
            else:
                for key,value in dotP_negW[2].items():
                    if key in W:
                        W[key] += value
                    else:
                        W[key] = value
        epochs+=1
    print("Training Accuracy:",(totalRows-timesUpdated)/totalRows)
    return [W, bias]

'''
Stuff
'''
def perceptron(data, r, bias):
    W = {} #Our weight vector is zero at first
    #For Debugging *******************
    timesUpdated = 0.0
    totalRows = float(len(data))
    #****************************
    #Decaying etta/1+t learning rate
    for row in data:
        label = int(row[0])
        #Update learning rate
        #Make a prediction based on the sign of (w_t(x_i))
        dotP_negW = wDotX(W, row, r,bias)
        prediction = sgnOf(dotP_negW[0])
        if prediction != label:
            timesUpdated += 1
            #Update bias
            bias += (r*label)
            #Update W vector 
            if label < 0:
                #We calculated the negW while doing the dot product
                #Add to W 
                for key,value in dotP_negW[1].items():
                    if key in W:
                        W[key] += value
                    else:
                        W[key] = value
            #Otherwise it's just the scaled version
            else:
                for key,value in dotP_negW[2].items():
                    if key in W:
                        W[key] += value
                    else:
                        W[key] = value
    print("Training Accuracy:", (totalRows-timesUpdated)/totalRows)
    return [W, bias]

def perceptronAvgReport20(param):
    accuracies = []
    vectors = []
    counter = 0
    train_test = catFilesForCV(100)
    W_bias = [{},{},0,0]
    while(counter < 20):
        random.shuffle(train_test[0])
        W_bias = perceptronAvgHW(train_test[0], param, W_bias[2], W_bias[0],W_bias[1], W_bias[3])
        cpy = copy.deepcopy(W_bias)
        vectors.append(cpy)
        accuracy = oracle(W_bias[0],train_test[1], W_bias[1])
        accuracies.append([counter, accuracy])
        counter+=1
    return [accuracies, vectors]

def perceptronAvgReport10(hyperparameters):
    top_accuracies = []
    counter = 0
    for i in range(0,5):
        train_test=catFilesForCV(i)
        local_max = 0.0
        bestParam = 0.0
        for param in hyperparameters:
            #10 epoch run
            W_bias = perceptronAvgHW(train_test[0], param, 0, {}, {}, 0)
            while(counter < 9):
                random.shuffle(train_test[0])
                W_bias = perceptronAvgHW(train_test[0], param, W_bias[2], W_bias[0],W_bias[1], W_bias[3])
                counter+=1
            counter = 0
            accuracy = oracle(W_bias[1],train_test[1], W_bias[3])
            if accuracy > local_max:
                local_max = accuracy
                bestParam = param
        top_accuracies.append([local_max, bestParam])
    return top_accuracies

def perceptronAvgReport():
    #####################
    #Basic Perceptron HW implementation (as oppossed to slides)
    #(dataset, rateOfLearning, bias, epoch_limit)
    #####################
    hyperparameters= [1, 0.1, 0.01]
    top_accuracies = perceptronAvgReport10(hyperparameters)
    print()
    print("##################################")
    print("Average Perceptron Report")
    print("##################################")
    print()
    print("----------------------------")
    print("10 epochs on each hyperparameter")
    print()
    print("Elements of type:\n                    [Best_Accuracy, Best_hyperparmeter]")
    scores = [0,0,0]
    for i in range(0,5):
        if(top_accuracies[i][1] == hyperparameters[0]):
            scores[0] += 1
        elif(top_accuracies[i][1] == hyperparameters[1]):
            scores[1] += 1
        elif(top_accuracies[i][1] == hyperparameters[2]):
            scores[2] += 1
        print( "Test Set:training0%d.data" %i, top_accuracies[i])
    print("----------------------------")
    print()
    max_value = max(scores)
    max_index = scores.index(max_value)
    best_hp = hyperparameters[max_index]

    #I will train the diabetes.train set for 20 epochs with this hp
    max_acc = [0, 0]
    results20 = perceptronBasicReport20(best_hp)
    for item in results20[0]:
        if(item[1] > max_acc[1]):
            max_acc[1] = item[1]
            max_acc[0] = item[0]


    print("----------------------------")
    print("20 epoch trials using best hyperparameter from above")
    print("[+] Best hyperparameter", best_hp)
    #get vector with @ epoch with maximum accuracy
    print("[+] Vector with most accuracy", max_acc)
    W20 = results20[1][max_acc[0]]
    
    #test the diabetes.test set with the W vector @ the epoch that has highest
    #accuracy
    test_set = []
    test_set = readFile('diabetes.test',test_set)
    accuracy = oracle(W20[0],test_set, W20[1])
    print("[+] Accuracy using W vector from epoch with most accuracy")
    print(accuracy)
    print("----------------------------")



'''
MARGIN
'''

def perceptronMarginReport20(param, mu):
    accuracies = []
    vectors = []
    counter = 0
    train_test = catFilesForCV(100)
    W_bias = [{},0]
    while(counter < 20):
        random.shuffle(train_test[0])
        W_bias = perceptronMarginHW(train_test[0], param, W_bias[1], W_bias[0], mu)
        cpy = copy.deepcopy(W_bias)
        vectors.append(cpy)
        accuracy = oracle(W_bias[0],train_test[1], W_bias[1])
        accuracies.append([counter, accuracy])
        counter+=1
    return [accuracies, vectors]


def perceptronMarginReport10(hyperparameters, margins):
    top_accuracies = []
    counter = 0
    for i in range(0,5):
        train_test=catFilesForCV(i)
        local_max = 0.0
        bestParam = 0.0
        for param in hyperparameters:
            for mu in margins:
                W_bias = perceptronMarginHW(train_test[0], param, 0, {}, mu)
                while(counter < 9):
                    random.shuffle(train_test[0])
                    W_bias = perceptronMarginHW(train_test[0], param, W_bias[1], W_bias[0], mu)
                    counter+=1
                counter = 0
                accuracy = oracle(W_bias[0],train_test[1], W_bias[1])
                if accuracy > local_max:
                    local_max = accuracy
                    bestParam = param
                top_accuracies.append([local_max, bestParam, mu])
    return top_accuracies



def perceptronMarginReport():
    #####################
    #Margin Perceptron HW implementation (as oppossed to slides)
    #(dataset, rateOfLearning, bias, epoch_limit)
    #####################
    hyperparameters= [1, 0.1, 0.01]
    top_accuracies = perceptronMarginReport10(hyperparameters, hyperparameters)
    print()
    print("##################################")
    print("Margin Perceptron Report")
    print("##################################")
    print()
    print("----------------------------")
    print("10 epochs on each hyperparameter")
    print()
    print("Elements of type:\n                    [Best_Accuracy, Best_hyperparmeter, Best Mu]")
    paramScores = [0,0,0]
    muScores = [0,0,0]
    for i in range(0,5):
        if(top_accuracies[i][1] == hyperparameters[0]):
            paramScores[0] += 1
        elif(top_accuracies[i][1] == hyperparameters[1]):
            paramScores[1] += 1
        elif(top_accuracies[i][1] == hyperparameters[2]):
            paramScores[2] += 1
        if(top_accuracies[i][2] == hyperparameters[0]):
            muScores[0] += 1
        elif(top_accuracies[i][1] == hyperparameters[1]):
            muScores[1] += 1
        elif(top_accuracies[i][1] == hyperparameters[2]):
            muScores[2] += 1
        print( "Test Set:training0%d.data" %i, top_accuracies[i])
    print("----------------------------")
    print()
    max_value = max(paramScores)
    max_index = paramScores.index(max_value)
    best_hp = hyperparameters[max_index]

    max_value = max(muScores)
    max_index = muScores.index(max_value)
    best_mu = hyperparameters[max_index]


    #I will train the diabetes.train set for 20 epochs with this hp
    max_acc = [0, 0]
    results20 = perceptronMarginReport20(best_hp, best_mu)
    for item in results20[0]:
        if(item[1] > max_acc[1]):
            max_acc[1] = item[1]
            max_acc[0] = item[0]


    print("----------------------------")
    print("20 epoch trials using best hyperparameter from above")
    print("[+] Best hyperparameter", best_hp)
    #get vector with @ epoch with maximum accuracy
    print("[+] Vector with most accuracy", max_acc)
    W20 = results20[1][max_acc[0]]
    
    #test the diabetes.test set with the W vector @ the epoch that has highest
    #accuracy
    test_set = []
    test_set = readFile('diabetes.test',test_set)
    accuracy = oracle(W20[0],test_set, W20[1])
    print("[+] Accuracy using W vector from epoch with most accuracy")
    print(accuracy)
    print("----------------------------")


'''
Decaying
'''

def perceptronDecayingReport20(param):
    accuracies = []
    vectors = []
    counter = 0
    train_test = catFilesForCV(100)
    W_bias = [{},0, 0]
    while(counter < 20):
        random.shuffle(train_test[0])
        W_bias = perceptronDecayingHW(train_test[0], param, W_bias[1], W_bias[0], W_bias[2])
        cpy = copy.deepcopy(W_bias)
        vectors.append(cpy)
        accuracy = oracle(W_bias[0],train_test[1], W_bias[1])
        accuracies.append([counter, accuracy])
        counter+=1
    return [accuracies, vectors]

def perceptronDecayingReport10(hyperparameters):
    top_accuracies = []
    counter = 0
    for i in range(0,5):
        train_test=catFilesForCV(i)
        local_max = 0.0
        bestParam = 0.0
        for param in hyperparameters:
            #10 epoch run
            W_bias = perceptronDecayingHW(train_test[0], param, 0, {}, 0)
            while(counter < 9):
                random.shuffle(train_test[0])
                W_bias = perceptronDecayingHW(train_test[0], param, W_bias[1], W_bias[0], W_bias[2])
                counter+=1
            counter = 0
            accuracy = oracle(W_bias[0],train_test[1], W_bias[1])
            if accuracy > local_max:
                local_max = accuracy
                bestParam = param
        top_accuracies.append([local_max, bestParam])
    return top_accuracies

def perceptronDecayingReport():
    #####################
    #Decaying Perceptron HW implementation (as oppossed to slides)
    #(dataset, rateOfLearning, bias, epoch_limit)
    #####################
    hyperparameters= [1, 0.1, 0.01]
    top_accuracies = perceptronDecayingReport10(hyperparameters)
    print()
    print("##################################")
    print("Decaying Perceptron Report")
    print("##################################")
    print()
    print("----------------------------")
    print("10 epochs on each hyperparameter")
    print()
    print("Elements of type:\n                    [Best_Accuracy, Best_hyperparmeter]")
    scores = [0,0,0]
    for i in range(0,5):
        if(top_accuracies[i][1] == hyperparameters[0]):
            scores[0] += 1
        elif(top_accuracies[i][1] == hyperparameters[1]):
            scores[1] += 1
        elif(top_accuracies[i][1] == hyperparameters[2]):
            scores[2] += 1
        print( "Test Set:training0%d.data" %i, top_accuracies[i])
    print("----------------------------")
    print()
    max_value = max(scores)
    max_index = scores.index(max_value)
    best_hp = hyperparameters[max_index]

    
    #I will train the diabetes.train set for 20 epochs with this hp
    max_acc = [0, 0]
    results20 = perceptronDecayingReport20(best_hp)
    for item in results20[0]:
        if(item[1] > max_acc[1]):
            max_acc[1] = item[1]
            max_acc[0] = item[0]


    print("----------------------------")
    print("20 epoch trials using best hyperparameter from above")
    print("[+] Best hyperparameter", best_hp)
    #get vector with @ epoch with maximum accuracy
    print("[+] Vector with most accuracy", max_acc)
    W20 = results20[1][max_acc[0]]
    
    #test the diabetes.test set with the W vector @ the epoch that has highest
    #accuracy
    test_set = []
    test_set = readFile('diabetes.test',test_set)
    accuracy = oracle(W20[0],test_set, W20[1])
    print("[+] Accuracy using W vector from epoch with most accuracy")
    print(accuracy)
    print("----------------------------")

 
'''
BASIC STUFF
'''
def plotLearningCurve(perceptronAlgorithm):
    train_test = catFilesForCV(99)
    accuracies = []
    epochs = []
    W_bias = [{},{},0,0]
    for i in range(1,20):
        W_bias = perceptronAlgorithm(train_test[0], 0.01, W_bias[2], W_bias[0],W_bias[1], W_bias[3])
        #W_bias = perceptronAlgorithm()
        accuracy = oracle(W_bias[0], train_test[1], W_bias[2])
        epochs.append(i)
        accuracies.append(accuracy)
    print(accuracies, epochs)
    plt.plot(epochs,accuracies)
    plt.show()


def perceptronBasicReport20(param):
    accuracies = []
    vectors = []
    counter = 0
    train_test = catFilesForCV(100)
    W_bias = [{},0]
    while(counter < 20):
        random.shuffle(train_test[0])
        W_bias = perceptronHW(train_test[0], param, W_bias[1], W_bias[0])
        cpy = copy.deepcopy(W_bias)
        vectors.append(cpy)
        accuracy = oracle(W_bias[0],train_test[1], W_bias[1])
        accuracies.append([counter, accuracy])
        counter+=1
    return [accuracies, vectors]

def perceptronBasicReport10(hyperparameters):
    top_accuracies = []
    counter = 0
    for i in range(0,5):
        train_test=catFilesForCV(i)
        local_max = 0.0
        bestParam = 0.0
        for param in hyperparameters:
            #10 epoch run
            W_bias = perceptronHW(train_test[0], param, 0, {})
            while(counter < 9):
                random.shuffle(train_test[0])
                W_bias = perceptronHW(train_test[0], param, W_bias[1], W_bias[0])
                counter+=1
            counter = 0
            accuracy = oracle(W_bias[0],train_test[1], W_bias[1])
            if accuracy > local_max:
                local_max = accuracy
                bestParam = param
        top_accuracies.append([local_max, bestParam])
    return top_accuracies

def perceptronBasicReport():
    #####################
    #Basic Perceptron HW implementation (as oppossed to slides)
    #(dataset, rateOfLearning, bias, epoch_limit)
    #####################
    hyperparameters= [1, 0.1, 0.01]
    top_accuracies = perceptronBasicReport10(hyperparameters)
    print()
    print("##################################")
    print("Basic Batch Perceptron Report")
    print("##################################")
    print()
    print("----------------------------")
    print("10 epochs on each hyperparameter")
    print()
    print("Elements of type:\n                    [Best_Accuracy, Best_hyperparmeter]")
    scores = [0,0,0]
    for i in range(0,5):
        if(top_accuracies[i][1] == hyperparameters[0]):
            scores[0] += 1
        elif(top_accuracies[i][1] == hyperparameters[1]):
            scores[1] += 1
        elif(top_accuracies[i][1] == hyperparameters[2]):
            scores[2] += 1
        print( "Test Set:training0%d.data" %i, top_accuracies[i])
    print("----------------------------")
    print()
    max_value = max(scores)
    max_index = scores.index(max_value)
    best_hp = hyperparameters[max_index]

    #I will train the diabetes.train set for 20 epochs with this hp
    max_acc = [0, 0]
    results20 = perceptronBasicReport20(best_hp)
    for item in results20[0]:
        if(item[1] > max_acc[1]):
            max_acc[1] = item[1]
            max_acc[0] = item[0]


    print("----------------------------")
    print("20 epoch trials using best hyperparameter from above")
    print("[+] Best hyperparameter", best_hp)
    #get vector with @ epoch with maximum accuracy
    print("[+] Vector with most accuracy", max_acc)
    W20 = results20[1][max_acc[0]]
    
    #test the diabetes.test set with the W vector @ the epoch that has highest
    #accuracy
    test_set = []
    test_set = readFile('diabetes.test',test_set)
    accuracy = oracle(W20[0],test_set, W20[1])
    print("[+] Accuracy using W vector from epoch with most accuracy")
    print(accuracy)
    print("----------------------------")

def catFilesForCV(num):
    DIR = 'CVSplits/'
    #erase the header of the concatenated files
    if(num == 0):
        data1 = []
        data2 = []
        data3 = []
        data4 = []
        test_set = []
        data1 = readFile(DIR+'training01.data',data1)
        data2 = readFile(DIR+'training02.data',data2)
        data3 = readFile(DIR+'training03.data',data3)
        data4 = readFile(DIR+'training04.data',data4)
        
        train_set = data1+data2+data3+data4
        test_set = readFile(DIR+'training00.data',test_set)
    elif(num == 1):
        data1 = []
        data2 = []
        data3 = []
        data4 = []
        test_set = []
        data1 = readFile(DIR+'training00.data',data1)
        data2 = readFile(DIR+'training02.data',data2)
        data3 = readFile(DIR+'training03.data',data3)
        data4 = readFile(DIR+'training04.data',data4)
        
        train_set = data1+data2+data3+data4
        test_set = readFile(DIR+'training01.data',test_set)
    elif(num == 2):
        data1 = []
        data2 = []
        data3 = []
        data4 = []
        test_set = []
        data1 = readFile(DIR+'training00.data',data1)
        data2 = readFile(DIR+'training01.data',data2)
        data3 = readFile(DIR+'training03.data',data3)
        data4 = readFile(DIR+'training04.data',data4)
        
        train_set = data1+data2+data3+data4
        test_set = readFile(DIR+'training02.data',test_set)
    elif(num == 3):
        data1 = []
        data2 = []
        data3 = []
        data4 = []
        test_set = []
        data1 = readFile(DIR+'training00.data',data1)
        data2 = readFile(DIR+'training01.data',data2)
        data3 = readFile(DIR+'training02.data',data3)
        data4 = readFile(DIR+'training04.data',data4)
        
        train_set = data1+data2+data3+data4
        test_set = readFile(DIR+'training03.data',test_set)
    elif(num == 4):
        data1 = []
        data2 = []
        data3 = []
        data4 = []
        test_set = []
        data1 = readFile(DIR+'training00.data',data1)
        data2 = readFile(DIR+'training02.data',data2)
        data3 = readFile(DIR+'training03.data',data3)
        data4 = readFile(DIR+'training01.data',data4)
        
        train_set = data1+data2+data3+data4
        test_set = readFile(DIR+'training04.data',test_set)
    elif(num == 99):
        train_set = []
        test_set = []
        train_set = readFile('diabetes.train',train_set)
        test_set = readFile('diabetes.test',test_set)
    elif(num == 100):
        train_set = []
        test_set = []
        train_set = readFile('diabetes.train',train_set)
        test_set = readFile('diabetes.dev',test_set)
    else:
        print("Argument not in range")
        
    return [train_set, test_set]


def readFile(filepath, data):
    with open(filepath, 'r') as fp:
        line = fp.readline()
        while (line):
            row = line.split()
            data.append(row)
            line = fp.readline()
        fp.close()
    return data

def countLabels(data):
    counter=[0,0]
    for line in data:
        if(int(line[0]) < 0):
            counter[0]+=1
        else:
            counter[1]+=1
    print(counter)

def main():
    print("*****************************************")
    print("Welcome to the Perceptron Driver")
    print("*****************************************")
    print()
    train_test = catFilesForCV(99)
#    train_dev = catFilesForCV(100)
    #plotLearningCurve(perceptronAvgHW)
    countLabels(train_test[0])
    #perceptronAvgHW(train_test[0], 0.01, 0, {}, {},0)
    W_bias = [{},0]
    W_bias = perceptronHW(train_test[0], 1, W_bias[1], W_bias[0])
    accuracyTest = majorityOracle(1, train_test[1])
   # accuracyDev = majorityOracle(1,train_dev[1])
    print(accuracyTest)
'''
    perceptronBasicReport()
    perceptronDecayingReport()
    perceptronMarginReport()
    perceptronAvgReport()
'''

if __name__=="__main__":
    main()
