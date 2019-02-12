import copy
import random
import math
import numpy as np
import time

'''
-666 = index of label
-555 = bias position
'''

DEPTH = 0

class Node:
    def __init__(self):
        self.depth = 0
        self.attribute = 0
        self.branches = {}
        self.label = 0
        self.leaf = False
#############################[DECISION TREES]############################
def cloneNode(rootNode):
    copyNode = Node()
    copyNode.attribute = rootNode.attribute
    copyNode.branches = rootNode.branches
    copyNode.label = rootNode.label
    copyNode.leaf = rootNode.leaf
    return copyNode
    
def id3_oracle(rootNode, testSet, dee):
    A_tpfpfn = [0,0,0,0]
    for row in testSet:
        node = cloneNode(rootNode)
        while(node.depth != dee):
            #if attribute in node is in the row, go down the 1 branch
            if(node.attribute in row):
                node = node.branches[1]
            else:
                node = node.branches[0]
        prediction = node.label
        print("pred", prediction)
        '''
        if(row[-666] == prediction):
            A_tpfpfn[0] += 1
        if(row[-666] == 1 and prediction == 1):
            A_tpfpfn[1] += 1
        #false positives
        elif(prediction >= 0 and row[-666] < 0):
            A_tpfpfn[2] += 1
        elif(prediction < 0 and row[-666] >= 0):
            A_tpfpfn[3] += 1
    A_tpfpfn[0] = A_tpfpfn[0]/len(testSet)
    return A_tpfpfn
'''
        
def make_entropy(p_plus, p_neg):
    if(p_plus == 0 or p_neg == 0):
        return 0
    else:
        H = -(p_plus)*math.log2(p_plus) - (p_neg) * math.log2(p_neg)
    return H

'''                
#specific to this dataset
def information_gain(global_entropy, global_size, attribute, subset):
    counts = count_feature_labels(subset, attribute)
    local_entropy_pos = make_entropy(count[0], counts[1])
    local_entropy_neg = make_entropy(count[1], counts[0])
    magnitude_sv = counts[0] + counts[1]
    summation = 
    pass
'''    
#specific to this dataset
def count_feature_labels(dataset, attribute):
    counts = [0,0]
    rows_that_have_it = 0
    for row in dataset:
        if(row[attribute] == 1):
            counts[0] += 1
        #in this dataset if it's not in the feature vector the value is zero
        else:
            counts[1] += 1
            
def count_labels(dataset, p_label, n_label):
    counts = [0,0]
    for row in dataset:
        if(row[-666] == p_label):
            counts[0] += 1
        elif(row[-666] == n_label):
             counts[1] += 1
        else:
             print("ERROR in counting labels")
    return counts

def calculate_information_gains(S, attributes, current_state_entropy):
    information_gains = {}
    max_ig = [0,0]
    for attribute in attributes:
        local_total = [0,0]
        one_proportions = [0,0]
        zero_proportions = [0,0]
        for row in S:
            #if(row[attribute] == 1):
            if(attribute in row):
                local_total[0] += 1
                if(row[-666] == 1):
                    one_proportions[0] += 1
                else:
                    one_proportions[1] += 1
            else:
                local_total[1] += 1
                if(row[-666] == 1):
                    zero_proportions[0] += 1
                else:
                    zero_proportions[1] += 1
        #Attribute was not in any row, so proportion is zero
        if(local_total[0] == 0):
            one_p_plus = 0
            one_p_neg  = 0
        else:
            one_p_plus = one_proportions[0]/local_total[0]
            one_p_neg  = one_proportions[1]/local_total[0]
        if(local_total[1] == 0):
            zed_p_plus = 0
            zed_p_neg = 0
        else:
            zed_p_plus = zero_proportions[0]/local_total[1]
            zed_p_neg  = zero_proportions[1]/local_total[1]
        one_entropy = make_entropy(one_p_plus, one_p_neg)
        zed_entropy = make_entropy(zed_p_plus, zed_p_neg)
        total = local_total[0] + local_total[1]
        IG = current_state_entropy - ( (local_total[0]/total)*one_entropy ) - ((local_total[1]/total) * zed_entropy)
        '''
        print("IG:",IG)
        print("CS:", current_state_entropy)
        print("lt:", local_total)
        print("att:", attribute)
        '''
        if(IG == 0):
            max_ig[0] = attribute
            max_ig[1] = IG
        elif(IG > max_ig[1]):
            max_ig[0] = attribute
            max_ig[1] = IG
       # information_gains[attribute] = IG
    return max_ig

def make_subset(S, attribute):
    one_set = []
    zed_set = []
    for row in S:
        if attribute in row:
            one_set.append(row)
        else:
            zed_set.append(row)
    return [one_set, zed_set]

def id3(S, label, attributes, dee):
    global DEPTH
    counts = count_labels(S, 1, -1)
    rootNode = Node()
    #all labels are negative
    if(counts[0] == 0):
        #add negative label
        rootNode.label = -1
        rootNode.leaf = True
        rootNode.depth = DEPTH
        return rootNode
    #all labels are positive
    elif(counts[1] == 0):
        #add negative label
        rootNode.label = 1
        rootNode.leaf = True
        rootNode.depth = DEPTH
        return rootNode
    else:
        #most common label 
        if(counts[0] > counts[1] or counts[0] == counts[1]):
            label = 1
        else:
            label = -1
        total_count = counts[0] + counts[1]
        rootNode.label = label
        rootNode.depth = DEPTH
        current_state_entropy = make_entropy(counts[0]/total_count, counts[1]/total_count)
        #print(current_state_entropy)
        #print(counts)
        #print(attributes)
        #calculate entropy with respect to each attribute
        if(len(attributes) == 0 or DEPTH == dee):
            return rootNode
        A = calculate_information_gains(S, attributes, current_state_entropy)
       # print(A[0])
        rootNode.attribute = A[0]
        #remove attribute from attributes
        attributes.remove(A[0])
       # print("LENAT:", len(attributes))
        subsets = make_subset(S, A[0])
        DEPTH += 1
        for i in range(0, 2):
            rootNode.branches[i] = Node()
            #if we are in 0 branch and it's subset is empty
            if(i == 0 and len(subsets[1]) == 0):
                rootNode.label = label
                rootNode.leaf = True
            elif(i == 1 and len(subsets[0]) == 0):
                rootNode.label = label
                rootNode.leaf = True
            else:
                if(i == 0):
                    rootNode.branches[i] = id3(subsets[1], label, attributes, dee)
                else:
                    rootNode.branches[i] = id3(subsets[0], label, attributes, dee)
                    
    return rootNode

#############################[LOGISTIC REGRESSION]############################
    
def logistic_regression(sigma, gamma, epochs, dataset):
    W = {}
    for i in range(0,epochs):
        for row in dataset:
            #Calculate yiWtXi
            exponent = row[-666] * wDotXi(W, row)
            denominator = np.exp(exponent) + 1
            numerator_vector = multiply_vector(row,row[-666])
            first_term = multiply_vector(numerator_vector, (1/denominator))
            sig_term = 2/(sigma**2)
            sig_vector = multiply_vector(W, sig_term)
            both_terms = add_vectors(first_term, sig_vector)
            gamma_vector = multiply_vector(both_terms, -gamma)
            W = add_vectors(W, gamma_vector)
        random.shuffle(dataset)
    return W
            
#
#############################[NAIVE BAYES]###################
#[+, -, log(+), log(-)]
#def log_probabilities
def prior(dataset):
    pos_neg=[0,0,0,0]
    for row in dataset:
        if(row[-666] == 1):
            pos_neg[0] += 1
        else:
            pos_neg[1] += 1
    pos_neg[2] = np.log(pos_neg[0]/len(dataset))
    pos_neg[3] = np.log(pos_neg[1]/len(dataset))
    return pos_neg

def get_219_attribute_count(dataset):
    #For some reason they start at # 2
    yes_dic = {}
    no_dic = {}
    for i in range(2,220):
        for row in dataset:
            #if the attribute exists in the row
            if(i in row):
                #if the label is pos
                if(row[-666] == 1):
                    if i in yes_dic:
                        yes_dic[i] += 1
                    else:
                        yes_dic[i] = 1
                elif(row[-666] == -1):
                    if i in no_dic:
                        no_dic[i] += 1
                    else:
                        no_dic[i] = 1
    return [yes_dic, no_dic]

def make_log_probabilities(yes, no, y_n_total, lamda, Si):
    n_yes = {}
    n_no = {}
    for key in yes:
        #yes[key] = np.log( (yes[key]+lamda)/(y_n_total[0] + Si * lamda))
        n_yes[key] = np.log( (yes[key]+lamda)/(y_n_total[0] + Si * lamda))
        #print(yes[key])
    for key in no:
        #no[key] = np.log(  (no[key] +lamda)/(y_n_total[1] + Si * lamda))
        n_no[key] = np.log(  (no[key] +lamda)/(y_n_total[1] + Si * lamda))
    return [n_yes, n_no]

##I'm not * by p(y), p(y)*P(x1...xj) NEED TO
def naive_predictor(yp_np, dataset, p):
    A_tpfpfn = [0,0,0,0]
    for row in dataset:
        pos_sum = 0.0
        neg_sum = 0.0
        prediction = 0
        label = 0
        for key in row:
            if(key == -666):
                label = row[key]
            if(key == -555):
                pass
            #Calculate pos prob
            if(key in yp_np[0]):
                pos_sum += yp_np[0][key]
            #Calculate neg prob
            if(key in yp_np[1]):
                neg_sum += yp_np[1][key]
        if((pos_sum+p[2]) > (neg_sum+p[3])):
            prediction = 1
        elif((pos_sum+p[2]) < (neg_sum+p[3])):
            prediction = -1
        # IF TIE DECIDE ONE ARITRARILY 
        else:
            prediction = 1
        if(prediction == label):
            A_tpfpfn[0] += 1
        if(label == 1 and prediction == label):
            A_tpfpfn[1] += 1
        elif(label == -1 and prediction == 1):
            A_tpfpfn[2] += 1
        elif(label == 1 and prediction == -1):
            A_tpfpfn[3] += 1
    A_tpfpfn[0] = A_tpfpfn[0]/len(dataset)
    return A_tpfpfn
        
            
        
        
#
#############################[SVM]#########################
#
def oracle(W, test_set):
    A_tpfpfn = [0, 0, 0, 0]
    for row in test_set:
        prediction = sgnOf(wDotXi(W, row))
        #if label == prediction
        if(row[-666] == prediction):
            A_tpfpfn[0] += 1
        if(row[-666] == 1 and prediction == 1):
            A_tpfpfn[1] += 1
        #false positives
        elif(prediction >= 0 and row[-666] < 0):
            A_tpfpfn[2] += 1
        elif(prediction < 0 and row[-666] >= 0):
            A_tpfpfn[3] += 1
    A_tpfpfn[0] = A_tpfpfn[0]/len(test_set)
    return A_tpfpfn

#counts = [accuracy, tp, fp,fn] 
def precision_recall_f(counts):
    #0/0 = 1
    if(counts[1] + counts[2] == 0):
        p = 1
    else:
        p = counts[1]/(counts[1] + counts[2])
    if(counts[1] + counts[3] == 0):
        r = 1
    else:
        r = counts[1]/(counts[1] + counts[3])
    if(p+r == 0):
        f = 1
    else:
        f = 2*((p*r)/(p+r))
    return [p, r, f]

             
def sgnOf(dotp):
    if (np.sign(dotp)== -1):
        return -1
    else:
        return 1
    

def wDotXi(W, X):
    suma = 0
    for key in X:
        if(key == -666): #skip the label
            pass
        elif(key in W):
            suma += X[key] * W[key]
    return suma

def multiply_vector(W, value):
    new_W = {}
    for key in W:
        if(key == -666):
            new_W[-666] = W[-666]
        else:
            new_W[key] = W[key] * value
    return new_W

#def multiply_list(row, value):
    
def add_vectors(W, X):
    new_W = {}
    for key in W:
        if(key == -666):
            new_W[-666] = W[-666]
        elif(key in X):
            new_W[key] = W[key] + X[key]
        else:
            new_W[key] = W[key]
            
    for key in X:
        #already added the values in common 
        if (key not in W):
            new_W[key] = X[key]
    return new_W
            
def SVM(W, data,gamma, C):
    #For each training example in data
    for row in data:
        label = row[-666]
        if(label*wDotXi(W,row) <= 1):
            W  = multiply_vector(W,1-gamma)
            Xi = multiply_vector(row, (gamma * C *label))
            W = add_vectors(W, Xi)
        else:
            W = multiply_vector(W, 1-gamma)
    return W

def SVM_epochs(epochs, train_test):
    #hyperparameters for SVM
    #gammas = [1, 0.1, 0.01, 0.001, 0.0001]
    #s = [1, 0.1, 0.01, 0.001, 0.0001]
    W = {}
    gammas = [ 0.01, 0.001]
    Cs=[100, 1000, 10000]
    for gamma in gammas:
        for C in Cs:
            for i in range(0, epochs):
                epoch_gamma = gamma/(1+i)
                W = SVM(W,train_test[0],gamma,C)
                random.shuffle(train_test[0])
            counts = oracle(W, train_test[1])
            prf    = precision_recall_f(counts)
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("         REPORT                        ")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("gamma=",gamma," C=",C)
            print(counts)
            print(prf)
            print()

    

#
#########################[SHARED]###############################
#
def catFilesForCV(num):
    DIR = 'CVSplits/'
    if(num == 0):
        t00 = []
        t01 = []
        t02 = []
        t03 = []
        t04 = []
        
        t00 = readFile(DIR+'training00.data',t00)
        t01 = readFile(DIR+'training01.data',t01)
        t02 = readFile(DIR+'training02.data',t02)
        t03 = readFile(DIR+'training03.data',t03)
        t04 = readFile(DIR+'training04.data',t04)
        
        train_set = t04 + t01 + t02 + t03 
        test_set = t00
    elif(num == 1):
        t00 = []
        t01 = []
        t02 = []
        t03 = []
        t04 = []
        
        t00 = readFile(DIR+'training00.data',t00)
        t01 = readFile(DIR+'training01.data',t01)
        t02 = readFile(DIR+'training02.data',t02)
        t03 = readFile(DIR+'training03.data',t03)
        t04 = readFile(DIR+'training04.data',t04)
        
        train_set = t00 + t04 + t02 + t03 
        test_set = t01
        
    elif(num == 2):
        t00 = []
        t01 = []
        t02 = []
        t03 = []
        t04 = []
        
        t00 = readFile(DIR+'training00.data',t00)
        t01 = readFile(DIR+'training01.data',t01)
        t02 = readFile(DIR+'training02.data',t02)
        t03 = readFile(DIR+'training03.data',t03)
        t04 = readFile(DIR+'training04.data',t04)
        
        train_set = t00 + t01 + t04 + t03 
        test_set = t02
    elif(num == 3):
        t00 = []
        t01 = []
        t02 = []
        t03 = []
        t04 = []
        
        t00 = readFile(DIR+'training00.data',t00)
        t01 = readFile(DIR+'training01.data',t01)
        t02 = readFile(DIR+'training02.data',t02)
        t03 = readFile(DIR+'training03.data',t03)
        t04 = readFile(DIR+'training04.data',t04)
        
        train_set = t00 + t01 + t02 + t04 
        test_set = t03
        
    elif(num == 4):
        t00 = []
        t01 = []
        t02 = []
        t03 = []
        t04 = []
        
        t00 = readFile(DIR+'training00.data',t00)
        t01 = readFile(DIR+'training01.data',t01)
        t02 = readFile(DIR+'training02.data',t02)
        t03 = readFile(DIR+'training03.data',t03)
        t04 = readFile(DIR+'training04.data',t04)
        
        train_set = t00 + t01 + t02 + t03 
        test_set = t04
    elif(num == 99):
        train_set = []
        test_set = []
        train_set = readFile('dummy.train',train_set)
        test_set = readFile('dummy.test',test_set)
    elif(num == 100):
        train_set = []
        test_set = []
        train_set = readFile('train.liblinear',train_set)
        test_set = readFile('test.liblinear',test_set)
    else:
        print("Argument not in range")
        
    return [train_set, test_set]


def dictionarize(row):
    dic = {}
    #label
    dic[-666] = int(row[0])
    #bias
    dic[-555] = 1
    for i in range(1, len(row)):
        elm = row[i]
        colon_idx = elm.index(':')
        key_idx = int(elm[:colon_idx])
        val_idx = float(elm[colon_idx+1:])
        dic[key_idx] = val_idx
    return dic

def readFile(filepath, data):
    with open(filepath, 'r') as fp:
        line = fp.readline()
        while (line):
            row = line.split()
            #need to make a dictionary of these values
            dic = dictionarize(row)
            data.append(dic)
            line = fp.readline()
        fp.close()
    return data

                



def main():
    print("Thanks for running me, although I'm named SVM.py\n all the code is in this one .py file")
    print("Please uncomment the code-block you'd like to run")
    print("Thank you!")
    
'''
    #For SVM CV
    for i in range(0, 5):
        train_test = catFilesForCV(i)
        print("[====================CVFOLD #",i, "=======================]")
        SVM_epochs(15, train_test)
'''
    
'''    
    global DEPTH
    dee = 30
    train_test = catFilesForCV(100)
    tree_vector = {}
    training = train_test[0]
    for i in range(0, 200):
        attributes = list(range(2, 220))
        random.shuffle(training)
        t = training[0:2000]
        rootNode = id3(t, 1, attributes, dee)
        DEPTH = 0
        print(rootNode.attribute)
        tree_vector[i] = rootNode

    #For SVM if my prediction of a tree is -1 don't put that into sparse vector
    nu_dataset = []
    for row in train_test[1]:
        nu_row = {}
        #bias
        nu_row[-555] = row[-555]
        nu_row[-666] = row[-666]
        for i in range(0, 200):
            rootN = tree_vector[i]
            node = cloneNode(rootN)
            while(node.depth != dee and node.attribute != 0):
                #if attribute in node is in the row, go down the 1 branch
                if(node.attribute in row):
                    node = node.branches[1]
                else:
                    node = node.branches[0]
            prediction = node.label
            if(prediction == -1):
                pass
            else:
                nu_row[i] = prediction
        nu_dataset.append(nu_row)
    
    for i in range(0,5):
        train_test = catFilesForCV(i)
        t_t = [nu_dataset, train_test[1]]
        print("[====================CVFOLD #",i, "=======================]")
        SVM_epochs(2, t_t)
'''
    

    

'''
    #Naive Bayes
    #NB SETUP
    for i in range(0, 5):
        print("====================[CV #",i,"]============================")
        train_test = catFilesForCV(i)
        p = prior(train_test[0])
        y_n = get_219_attribute_count(train_test[0])

        lamda = [2,1.5, 1.0, 0.5]
        for lam in lamda:
            print(lam)
            #yes, no, prior, lambda, Si
            yp_np = make_log_probabilities(y_n[0], y_n[1], p, lam, 2)
            results = naive_predictor(yp_np, train_test[1], p)
            prf    = precision_recall_f(results)
            print(results)
            print(prf)
    #Prediction
'''    
'''
    #Logistic Regression
    train_test = catFilesForCV(100)
    sigmas = [10, 100]
    gammas = [0.1, 0.001]
    epochs = 2
    for sigma in sigmas:
        for gamma in gammas:
            print("Sigma:", sigma)
            print("Gamma:", gamma)
            W = logistic_regression(sigma, gamma, epochs, train_test[0])
            counts = oracle(W, train_test[1])
            prf = precision_recall_f(counts)
            print(counts)
            print(prf)

'''

    



            

    



        

if __name__=="__main__":
    main()
