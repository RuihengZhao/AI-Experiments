from __future__ import division
from sets import Set
from math import log, fabs
import heapq
import copy

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

    wordsSet = []
    with open('words.txt', 'r') as w:
        for l in w:
            words.add(wordsCount)
            wordsSet.append(l.rstrip())
            wordsCount += 1

    return (wordsCount, words, wordsSet)

def predict(doc, atheismPrior, graphicsPrior, atheismLikelihoods, graphicsLikelihoods):
    l = len(atheismLikelihoods)
    for i in range(0, l):
        if i == 0:
            atheismPost = log(atheismPrior)
            graphicsPost = log(graphicsPrior)

        if doc[i]:
            atheismPost += log(atheismLikelihoods[i])
            graphicsPost += log(graphicsLikelihoods[i])
        else:
            atheismPost += log(1 - atheismLikelihoods[i])
            graphicsPost += log(1 - graphicsLikelihoods[i])
    
    if atheismPost >= graphicsPost:
        return 1
    else:
        return 2 

def accuracy(docs, atheismPrior, graphicsPrior, atheismLikelihoods, graphicsLikelihoods):
    numCorrectPredictions = 0
    for doc in docs:
        if predict(doc, atheismPrior, graphicsPrior, atheismLikelihoods, graphicsLikelihoods) == doc[-1]:
            numCorrectPredictions += 1

    return numCorrectPredictions/len(docs) * 100

def printWords(atheismLikelihoods, graphicsLikelihoods, wordsSet):
    res = []
    for i in range(0, len(atheismLikelihoods)):
        prLabel1 = log(atheismLikelihoods[i], 2)
        prLabel2 = log(graphicsLikelihoods[i], 2)
        decrimination = fabs(prLabel1 - prLabel2)
        res.append((i, decrimination))
    
    res.sort(lambda x,y: cmp(x[1], y[1]), None, True)
    for i in range(10):
        print 'word: {:<10}  ->  {}'.format(wordsSet[res[i][0]], res[i][1])

    return True

def learner(trainDataSparse, words):
    atheismCount = filter(lambda doc: doc[-1] == 1, trainDataSparse)
    graphicsCount = filter(lambda doc: doc[-1] == 2, trainDataSparse)

    trainDataSparseLen = len(trainDataSparse)
    labeledAtheismLen = len(atheismCount)
    labeledGraphicsLen = len(graphicsCount)

    atheismPrior = labeledAtheismLen / trainDataSparseLen
    graphicsPrior = labeledGraphicsLen / trainDataSparseLen

    atheismLikelihoods = []
    graphicsLikelihoods = []
    for word in words:
        # Laplace correction
        atheismNumerator = reduce(lambda x, y: x + (1 if y[word] else 0), atheismCount, 0)
        graphicsNumerator = reduce(lambda x, y: x + (1 if y[word] else 0), graphicsCount, 0)

        atheismLikelihoods.append((atheismNumerator + 1) / (labeledAtheismLen + 2))
        graphicsLikelihoods.append((graphicsNumerator + 1) / (labeledGraphicsLen + 2))

    return (atheismPrior, graphicsPrior, atheismLikelihoods, graphicsLikelihoods)

def main():
    wordsCount, words, wordsSet = loadWords()

    trainDataSparse = createSparse('trainData.txt', 'trainLabel.txt', wordsCount)
    testDataSparse = createSparse('testData.txt', 'testLabel.txt', wordsCount)

    atheismPrior, graphicsPrior, atheismLikelihoods, graphicsLikelihoods = learner(trainDataSparse, words)

    trainingAccuracy = accuracy(trainDataSparse, atheismPrior, graphicsPrior, atheismLikelihoods, graphicsLikelihoods)
    testAccuracy = accuracy(testDataSparse, atheismPrior, graphicsPrior, atheismLikelihoods, graphicsLikelihoods)

    print 'Training Accuracy: {:.3f}'.format(trainingAccuracy)
    print 'Testing Accuracy: {:.3f}'.format(testAccuracy)
    print ('\r')

    printDone = printWords(atheismLikelihoods, graphicsLikelihoods, wordsSet)

main()
