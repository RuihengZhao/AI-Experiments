from __future__ import division
from sets import Set
from math import log
import heapq
import copy

class DecisionTreeNode(object):
    def __init__(self, sparseMatrix, word, words, infoGain):
        self.sparseMatrix = sparseMatrix
        self.word = word
        self.words = copy.deepcopy(words)
        self.infoGain = infoGain
        self.pointEstimate = DecisionTreeNode.pointEst(sparseMatrix)

        self.rightChild = None
        self.leftChild = None

    def __cmp__(self, other):
        return -cmp(self.infoGain, other.infoGain)

    @staticmethod
    def pointEst(sparseMatrix):
        total = len(sparseMatrix)

        atheism = 0
        for i in range(0, total):
            if sparseMatrix[i][-1] == 1:
                atheism += 1

        graphics = total - atheism
        
        if atheism > graphics:
            return 1
        else:
            return 2

    @staticmethod
    def infoContent(sparseMatrix):
        N = len(sparseMatrix)

        if N == 0:
            return 1

        nAtheism = 0
        for i in range(0, N):
            if sparseMatrix[i][-1] == 1:
                nAtheism += 1
        
        nGraphics = N - nAtheism

        pAtheism = nAtheism / N
        pGraphics = nGraphics / N

        if pAtheism != 0 and pGraphics != 0:
            return (-pAtheism * log(pAtheism, 2)) + (-pGraphics * log(pGraphics, 2))
        else: 
            return 0

    @staticmethod
    def createNode(IE, sparseMatrix, words):
        maxIG = None
        bestWord = None

        for word in words:
            S1 = filter(lambda row: row[word] == 1, sparseMatrix)
            S2 = filter(lambda row: row[word] == 0, sparseMatrix)

            IE1 = DecisionTreeNode.infoContent(S1)
            IE2 = DecisionTreeNode.infoContent(S2)

            N1 = len(S1)
            N2 = len(S2)
            N = N1 + N2
            IEs = ((N1 / N) * IE1) + ((N2 / N) * IE2)

            IG = IE - IEs
            
            if maxIG is None or  IG > maxIG:
                maxIG = IG
                bestWord = word

        words.remove(bestWord)
        return DecisionTreeNode(sparseMatrix, bestWord, words, maxIG)

    def explore(self):
        if self.infoGain <= 0:
            return (None,None)
        
        S1 = filter(lambda row: row[self.word] == 1, self.sparseMatrix)
        S2 = filter(lambda row: row[self.word] == 0, self.sparseMatrix)

        IE1 = DecisionTreeNode.infoContent(S1)
        IE2 = DecisionTreeNode.infoContent(S2)

        self.rightChild = DecisionTreeNode.createNode(IE1, S1, self.words)
        self.leftChild = DecisionTreeNode.createNode(IE2, S2, self.words)

        return (self.rightChild, self.leftChild)

def createSparse(dataFile, labelFile, wordsCount):
    sparseMatrix = []

    with open(labelFile, 'r') as f:
        docId = 0
        for l in f:
            sparseMatrix.append([0] * (wordsCount + 1))
            sparseMatrix[docId][-1] = int(l)
            docId += 1
    
    with open(dataFile, 'r') as f:
        for l in f:
            lineSplit = l.split()
            docId = int(lineSplit[0]) - 1
            wordId = int(lineSplit[1]) - 1

            sparseMatrix[docId][wordId] = 1

    return sparseMatrix

def loadWords():
    wordsCount = 0
    words = []
    words = Set(words)

    with open('words.txt', 'r') as w:
        for l in w:
            words.add(wordsCount)
            wordsCount += 1

    return (wordsCount, words)

def predict(tree, doc):
    while True:
        if tree.rightChild and doc[tree.word] == 1 :
            tree = tree.rightChild
        elif tree.leftChild and doc[tree.word] == 0 :
            tree = tree.leftChild
        else:
            return tree.pointEstimate 

def accuracy(tree, matrix):
    correctPredict = 0

    for doc in matrix:
        prediction = predict(tree, doc)

        if prediction == doc[-1]:
            correctPredict += 1

    return correctPredict/len(matrix) * 100

def learner(decisionTree):
    priorityQueue = [decisionTree]

    for i in range(0, 100):
        leftChild, rightChild = heapq.heappop(priorityQueue).explore()

        if rightChild and leftChild:
            heapq.heappush(priorityQueue, rightChild)
            heapq.heappush(priorityQueue, leftChild)
        
        yield decisionTree

def main():
    wordsCount, words = loadWords()

    trainDataSparse = createSparse('trainData.txt', 'trainLabel.txt', wordsCount)
    testDataSparse = createSparse('testData.txt', 'testLabel.txt', wordsCount)

    IE = DecisionTreeNode.infoContent(trainDataSparse)
    root = DecisionTreeNode.createNode(IE, trainDataSparse, words)

    decisionTrees = learner(root)

    print ('Information gain weighted by the fraction of documents on each side of the split\r')
    print ('\r')
    print('Node#  Train Accuracy  Test Accuracy\r')

    for index, tree in enumerate(decisionTrees):
        trainAccuracy = accuracy(tree, trainDataSparse)
        testAccuracy = accuracy(tree, testDataSparse)

        print('{:<11} {:<15.3f} {:<.3f}'.format((index + 1), trainAccuracy, testAccuracy))

main()
