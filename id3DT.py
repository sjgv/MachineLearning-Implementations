import math
import statistics
import copy
from data import Data
import numpy as np
DATA_DIR = 'data/'
CV_DIR = 'data_new/CVfolds_new/'

data = np.loadtxt(DATA_DIR + 'train.csv', delimiter=',', dtype = str)
#data = np.loadtxt(DATA_DIR + 'mangos.csv', delimiter=',', dtype = str)
data_obj = Data(data = data)
data_obj2 = Data(fpath = DATA_DIR + 'test.csv')
#data_obj2 = Data(fpath = DATA_DIR + 'mangosTest.csv')
#TOTAL_ROWS = len(data_obj)

DEPTHS = []
##################[MY CODE]##############
class Node:
    def __init__(self):
        self.feature = ""
        self.features = set([])
        self.subset = []
        self.branches= {}
        self.label = ""
        self.counter = 0

def id3(dobj, attributes, label, level, depth):
    check = checkOneLabel(dobj, attributes)
    #Check most common label 
    if(check[1][0] > check[1][1]):
        label = "e"
    elif(check[1][1] > check[1][0]):
        label = "p"
    #if they're equal to each other or both are zero then choose one
    else:
        label = "e"

    #If Sv has same label     
    if(check[0]):
        n = Node()
        n.label = label
        #try incrementing level here as well
        level +=1
        return n
    else:
        rootNode = Node()
        best_classifying_feature_string = getRootNode(getInformationGain(dobj,attributes))[0]
        rootNode.feature = best_classifying_feature_string
        #line below not from the algorithm
        level += 1
        for v in attributes[best_classifying_feature_string].possible_vals:
            rootNode.branches[v] = Node()
            Sv = dobj.get_row_subset(best_classifying_feature_string, v)
            if(len(Sv) == 0):
                rootNode.branches[v].label = label
                #might not do anything
                rootNode.subset = [label]
            else:
                #Add most common label at each node to the node 
                rootNode.branches[v].label = label
                if(level == depth):
                    continue
                if(len(attributes) > 1 ):
                    copy_dic = attributes.copy()
                    copy_dic.pop(best_classifying_feature_string)
                    rootNode.branches[v] = id3(Sv, copy_dic, label, level, depth)
        return rootNode

'''
Returns a 
'''    
def checkOneLabel(subset, attributes):
    FLAG = ''
    first_label = True
    first_change = True
    ret_val = True
    counter = [0,0]
    #Choose a feature,the union of columns corresponding to possible values of that feature will be the entire set
    feature = list(attributes.keys())[0]
    #print(feature)
    for value in attributes[feature].possible_vals:
        zet = subset.get_row_subset(feature, value)
        for lbl in zet.get_column('label'):
            if(lbl == 'e'):
                counter[0]+=1
            elif(lbl == 'p'):
                counter[1]+=1
        #Check if one label is zero
        if(counter[0] == 0 and counter[1] > 0):
            pass
        elif(counter[1] == 0 and counter[0] > 0):
            pass
        else:
            ret_val = False
    return [ret_val, counter]

'''
Returns highest value of heuristic
requires: information gains type: [ [feature, heuristic], [...] ]
'''
def getRootNode(information_gain):
    top =[0.0, 0]
    for i in range(len(information_gain)):
        f = information_gain[i][1]
        if f > top[0]:
            top[0] = f
            top[1] = i
    return information_gain[top[1]]

'''
Returns information gain list of tuplets type: [ [feature, (globalEntropy - featureEntropy)], [...] ]
'''
def getInformationGain(subset,attributes):
    IG = []
    globalH = GLOBAL_ENTROPY
    #H is in pos 1 of tup
    Hs = getExpectedEntropies(subset, attributes)
    for h in Hs:
        IG.append([h[0], globalH - h[1]])
    return IG

'''
Returns global entropy type: int
'''
def getGlobalEntropy(data):
    Sv = []
    H = 0
    for i in range(len(data.raw_data)):
        Sv.append(data.raw_data[i][0])
    proportions = countEorP(Sv)
    pPlus = proportions[0]/len(data.raw_data)
    pMinus = proportions[1]/len(data.raw_data)
    #Check if float is zero
    if(pPlus < 0.0000000001):
        if(pMinus < 0.0000000001):
            pass
        else:
            H = -1 * (pMinus * math.log2(pMinus))
    else:
        if(pMinus < 0.0000000001):
            H = -1 * (pPlus * math.log2(pPlus))
        else:
            H = (-1 * (pPlus * math.log2(pPlus))) - (pMinus * math.log2(pMinus))
    return H 
    

'''
Returns a list of tuplets type: [feature, expected_entropy]
'''    
def getExpectedEntropies(subset, attributes):
    expected_entropies = []
    for feature in attributes:
        feature_Hs = []
        for feature_value in attributes[feature].possible_vals:
            H = 0
            data_subset = subset.get_row_subset(feature, feature_value)
            if(len(data_subset) > 0):
                labels = data_subset.get_column('label')
                subset_proportions = countEorP(labels)
                edible = subset_proportions[0]
                poisonous = subset_proportions[1]
                subset_total = edible + poisonous
                if(subset_total != 0):
                    pPlus = edible/subset_total
                    pMinus = poisonous/subset_total
                    P = subset_total/TOTAL_ROWS
                    #Check if float is zero 
                    if(pPlus < 0.0000000001):
                        if(pMinus < 0.0000000001):
                            t = [P,H]
                            feature_Hs.append(t)
                        else:
                            H = -1 * (pMinus * math.log2(pMinus))
                            t = [P,H]
                            feature_Hs.append(t)
                    else:
                        if(pMinus < 0.0000000001):
                            H = -1 * (pPlus * math.log2(pPlus))
                            t = [P,H]
                            feature_Hs.append(t)
                        else:
                            H = (-1 * (pPlus * math.log2(pPlus))) - (pMinus * math.log2(pMinus))
                            t = [P,H]
                            feature_Hs.append(t)
                else:
                    continue
            else:
                feature_Hs.append([0,0])
        expected_entropy=0
        #Multiply (proportion * H) for every value in feature
        #and save in expected_entropies
        for tup in feature_Hs:
            expected_entropy += tup[0]*tup[1]
        entry = [feature, expected_entropy]
        expected_entropies.append(entry)
    return expected_entropies
            
        


def countEorP(labels):
    results = [0,0]
    for result in labels:
        if result == 'e':
            results[0]+=1
        elif result == 'p':
            results[1]+=1
        else:
            print("This shouldn't be happenning")
    return results


def getDepthRecursive(node, data):
    if(node.feature == ""):
        #if(node.counter == 9):
         #   print(node.features)
        DEPTHS.append(node.counter)
        return
    else:
        node.counter += 1
        node.features.add(node.feature)
        for val in  data.attributes[node.feature].possible_vals:
            nuNode = node.branches[val]
            nuNode.counter = node.counter
            nuNode.features = node.features
            getDepthRecursive(nuNode, data)
      
                        

def cloneNode(rootNode):
    copyNode = Node()
    copyNode.feature = rootNode.feature
    copyNode.subset = copy.deepcopy(rootNode.subset)
    copyNode.branches= rootNode.branches.copy()
    copyNode.label = rootNode.label
    return copyNode

def traverse(line, rootNode, data):
    node = cloneNode(rootNode)
    while(node.feature != ""):
        #Add 1 to index of attribute here because line has 'label' at 0th pos
        feature_idx = data.attributes[node.feature].index + 1
        branch_to_travel = line[feature_idx]
        #A training set might not have features that test set does, so give label if not
        if branch_to_travel in node.branches:
            node = node.branches[branch_to_travel]
        else:
            return node.label
    return node.label

def getAccuracy(data, rootNode):
    counter = [0,0]
    #Get accuracy
    for line in data.raw_data:
        prediction_result = traverse(line, rootNode, data)
        #print("Prediction = " + prediction_result + " " + "Value= " + line[0])
        if(prediction_result == line[0]):
            counter[0]+=1
        else:
            counter[1]+=1
    total = counter[0]+counter[1]
    return (counter[0]/total)

def getMean(accuracies):
    mean = 0
    total_items = len(accuracies)
    for flt in accuracies:
        mean += flt
    mean = mean/total_items
    return mean

'''
Calculates stdDev of sample, not population
'''
def getStd(accuracies):
    return statistics.stdev(accuracies)

'''
Same as above but implemented by me
'''
def getStdMine(accuracies, mean):
    sqr_diffs = []
    sqr_diff = 0
    variance = 0
    #For each num: subtract mean and square result
    for flt in accuracies:
        sqr_diff = mean - flt
        sqr_diff = sqr_diff * sqr_diff
        sqr_diffs.append(sqr_diff)
        
    for flt in sqr_diffs:
        variance += flt
    variance = variance/(len(sqr_diffs)-1)
    std = math.sqrt(variance)
    return std
    
def crossValidation(depth):
    global TOTAL_ROWS
    global GLOBAL_ENTROPY
    accuracies = []
    for i in range(1,6):
        train_test = catFilesForCV(i)        
        #Get global entropy
        TOTAL_ROWS = len(train_test[0])
        GLOBAL_ENTROPY = getGlobalEntropy(train_test[0])
        rootNode = id3(train_test[0], train_test[0].attributes, 'e', 0, depth)

        #Save all the accuracies from same depth, diff CVfold
        accuracies.append(getAccuracy(train_test[1], rootNode))
    mean = getMean(accuracies)
    std = getStd(accuracies)
    return [mean, std]


def getCvOfDepths(depths):
    for depth in depths:
        print ("depth: " + str(depth),crossValidation(depth))


def singleRun(train_data, test_data, depth):
    global GLOBAL_ENTROPY
    global TOTAL_ROWS
    TOTAL_ROWS = len(train_data)
    GLOBAL_ENTROPY = getGlobalEntropy(train_data)
    rootNode = id3(train_data,train_data.attributes,'e',0,depth)
    counter = [0,0]
    train_acc = getAccuracy(train_data, rootNode)
    test_acc = getAccuracy(test_data,rootNode)
    
    getDepthRecursive(rootNode, test_data)
    max_depth = 0
    for depth in DEPTHS:
        if (depth > max_depth):
            max_depth = depth


    print("data/train.csv accuracy = " + str(train_acc))
    print("data/test.csv  accuracy = " + str(test_acc))
    print("Max depth               = " + str(max_depth))
    


        
def catFilesForCV(num):
    #erase the header of the concatenated files
    if(num == 1):
        data1 = np.loadtxt(CV_DIR + 'fold2.csv', delimiter=',', dtype = str)
        data2 = np.loadtxt(CV_DIR + 'fold3.csv', delimiter=',', dtype = str)
        data3 = np.loadtxt(CV_DIR + 'fold4.csv', delimiter=',', dtype = str)
        data4 = np.loadtxt(CV_DIR + 'fold5.csv', delimiter=',', dtype = str)
        data = np.concatenate((data1,data2[1:],data3[1:],data4[1:]))

        train_set = Data(data = data)
        test_set = Data(fpath = CV_DIR + 'fold1.csv')
    elif(num == 2):
        data1 = np.loadtxt(CV_DIR + 'fold1.csv', delimiter=',', dtype = str)
        data2 = np.loadtxt(CV_DIR + 'fold3.csv', delimiter=',', dtype = str)
        data3 = np.loadtxt(CV_DIR + 'fold4.csv', delimiter=',', dtype = str)
        data4 = np.loadtxt(CV_DIR + 'fold5.csv', delimiter=',', dtype = str)
        data = np.concatenate((data1,data2[1:],data3[1:],data4[1:]))

        train_set = Data(data = data)
        test_set = Data(fpath = CV_DIR + 'fold2.csv')
    elif(num == 3):
        data1 = np.loadtxt(CV_DIR + 'fold2.csv', delimiter=',', dtype = str)
        data2 = np.loadtxt(CV_DIR + 'fold1.csv', delimiter=',', dtype = str)
        data3 = np.loadtxt(CV_DIR + 'fold4.csv', delimiter=',', dtype = str)
        data4 = np.loadtxt(CV_DIR + 'fold5.csv', delimiter=',', dtype = str)
        data = np.concatenate((data1,data2[1:],data3[1:],data4[1:]))

        train_set = Data(data = data)
        test_set = Data(fpath = CV_DIR + 'fold3.csv')
    elif(num == 4):
        data1 = np.loadtxt(CV_DIR + 'fold2.csv', delimiter=',', dtype = str)
        data2 = np.loadtxt(CV_DIR + 'fold3.csv', delimiter=',', dtype = str)
        data3 = np.loadtxt(CV_DIR + 'fold1.csv', delimiter=',', dtype = str)
        data4 = np.loadtxt(CV_DIR + 'fold5.csv', delimiter=',', dtype = str)
        data = np.concatenate((data1,data2[1:],data3[1:],data4[1:]))

        train_set = Data(data = data)
        test_set = Data(fpath = CV_DIR + 'fold4.csv')
    elif(num == 5):
        data1 = np.loadtxt(CV_DIR + 'fold2.csv', delimiter=',', dtype = str)
        data2 = np.loadtxt(CV_DIR + 'fold3.csv', delimiter=',', dtype = str)
        data3 = np.loadtxt(CV_DIR + 'fold4.csv', delimiter=',', dtype = str)
        data4 = np.loadtxt(CV_DIR + 'fold1.csv', delimiter=',', dtype = str)
        data = np.concatenate((data1,data2[1:],data3[1:],data4[1:]))

        train_set = Data(data = data)
        test_set = Data(fpath = CV_DIR + 'fold5.csv')
    else:
        print("Argument not in range")
        
    return [train_set, test_set]

        
def main():
    print("=======================================================")
    print("    Starting Decision-Tree for Mushroom Data Set")
    print("=======================================================")
    
    depths = [1,2,3,4,5,7,10,15]
    singleRun(data_obj, data_obj2, 10)
#    getCvOfDepths(depths)
    
if __name__=="__main__":
    main()
