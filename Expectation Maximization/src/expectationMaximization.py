from __future__ import division
from functools import reduce
from math import sqrt
import random
import numpy as np

def addWhiteNoise(CPTs, delta):
    # CPTs with ramdom white noise between [0, delta)
    whitenoiseCPTs = [[[0, 0, 0],\
                       [0, 0, 0]],\
                      [0, 0, 0],\
                      [0, 0, 0],
                      [0],\
                      [0, 0]]

    for tableIndex in range(0, 5):
        r1 = random.uniform(0, delta)
        r2 = random.uniform(0, delta)

        if tableIndex == 0:
            for THTS in range(0, 2):
                for DS in range(0, 3):
                    whitenoiseCPTs[tableIndex][THTS][DS] = (CPTs[tableIndex][THTS][DS] + r1) / (1 + r1 + r2)

        elif (tableIndex == 1) or (tableIndex == 2):
            for DS in range(0, 3):
                whitenoiseCPTs[tableIndex][DS] = (CPTs[tableIndex][DS] + r1) / (1 + r1 + r2)

        elif tableIndex == 3:
            whitenoiseCPTs[tableIndex][0] = (CPTs[tableIndex][0] + r1) / (1 + r1 + r2)

        elif tableIndex == 4:
            for DS in range(0, 2):
                r3 = random.uniform(0, delta)

                whitenoiseCPTs[tableIndex][DS] = (CPTs[tableIndex][DS] + r1) / (1 + r1 + r2 + r3)

    return whitenoiseCPTs


def predict(testData, whitenoiseCPTs):
    numCorrect = 0

    for data in testData:
        S = data[0]
        F = data[1]
        D = data[2]
        THTS = data[3]
        DS = data[4]

        noneLikelihood = 1
        mildLikelihood = 1
        severeLikelihood = 1

        if S == 1:
            noneLikelihood *= whitenoiseCPTs[0][THTS][0]
            mildLikelihood *= whitenoiseCPTs[0][THTS][1]
            severeLikelihood *= whitenoiseCPTs[0][THTS][2]
        else:
            noneLikelihood *= (1 - whitenoiseCPTs[0][THTS][0])
            mildLikelihood *=  (1 - whitenoiseCPTs[0][THTS][1])
            severeLikelihood *= (1 - whitenoiseCPTs[0][THTS][2])
            
        if F == 1:
            noneLikelihood *= whitenoiseCPTs[1][0]
            mildLikelihood *= whitenoiseCPTs[1][1]
            severeLikelihood *= whitenoiseCPTs[1][2]
        else:
            noneLikelihood *= (1 - whitenoiseCPTs[1][0])
            mildLikelihood *=  (1 - whitenoiseCPTs[1][1])
            severeLikelihood *= (1 - whitenoiseCPTs[1][2])
            
        if D == 1:
            noneLikelihood *= whitenoiseCPTs[2][0]
            mildLikelihood *= whitenoiseCPTs[2][0]
            severeLikelihood *= whitenoiseCPTs[2][0]
        else:
            noneLikelihood *= (1 - whitenoiseCPTs[2][0])
            mildLikelihood *=  (1 - whitenoiseCPTs[2][0])
            severeLikelihood *= (1 - whitenoiseCPTs[2][0])

        if THTS == 1:
            noneLikelihood *= whitenoiseCPTs[3][0]
            mildLikelihood *= whitenoiseCPTs[3][0]
            severeLikelihood *= whitenoiseCPTs[3][0]
        else:
            noneLikelihood *= (1 - whitenoiseCPTs[3][0])
            mildLikelihood *=  (1 - whitenoiseCPTs[3][0])
            severeLikelihood *= (1 - whitenoiseCPTs[3][0])
            
        noneLikelihood *= whitenoiseCPTs[4][0]
        mildLikelihood *= whitenoiseCPTs[4][1]
        severeLikelihood *= (1 - whitenoiseCPTs[4][0] - whitenoiseCPTs[4][1])


        if (noneLikelihood >= mildLikelihood) and (noneLikelihood >= severeLikelihood):
            prediction = 0
        elif (mildLikelihood >= noneLikelihood) and (mildLikelihood >= severeLikelihood):
            prediction = 1
        else:
            prediction = 2

        if prediction == data[-1]:
            numCorrect += 1

    return numCorrect / len(testData)


def newJPLLTable(whitenoiseCPTs):
    # [S F D THTS DS] = [JointProbability Likelihood]
    JPLLTable = {}

    for S in range(0, 2):
        for F in range(0, 2):
            for D in range(0, 2):
                for THTS in range(0, 2):
                    sumJP = 0
                    JPs = []

                    for DS in range(0, 3):

                        JP = 1

                        if S == 1: 
                            JP *= whitenoiseCPTs[0][THTS][DS]
                        else:
                            JP *= (1 - whitenoiseCPTs[0][THTS][DS])

                        if F == 1: 
                            JP *= whitenoiseCPTs[1][DS]
                        else:
                            JP *= (1 - whitenoiseCPTs[1][DS])

                        if D == 1: 
                            JP *= whitenoiseCPTs[2][DS]
                        else:
                            JP *= (1 - whitenoiseCPTs[2][DS])

                        if THTS == 1: 
                            JP *= whitenoiseCPTs[3][0]
                        else:
                            JP *= (1 - whitenoiseCPTs[3][0])

                        if (DS == 0) or (DS == 1): 
                            JP *= whitenoiseCPTs[4][DS]
                        else:
                            JP *= (1 - whitenoiseCPTs[4][0] - whitenoiseCPTs[4][1])

                        JPs.append(JP)
                        sumJP += JP

                    for DS in range(0, 3):
                        JPLLTable[str([S, F, D, THTS, DS])] = [JPs[DS], JPs[DS]/sumJP]

    return JPLLTable


def sumLikelihood(data, likelihoodVal, sumAllLikelihood, sumThisLikelihood):
    S = data[0]
    F = data[1]
    D = data[2]
    THTS = data[3]
    DS = data[4]

    if S == 1:
        sumThisLikelihood[0][THTS][DS] += likelihoodVal

    if F == 1:
        sumThisLikelihood[1][DS] += likelihoodVal

    if D == 1:
        sumThisLikelihood[2][DS] += likelihoodVal

    if THTS == 1:
        sumThisLikelihood[3][0] += likelihoodVal

    if (DS == 0) or (DS == 1):
        sumThisLikelihood[4][DS] += likelihoodVal

    sumAllLikelihood[0][THTS][DS] += likelihoodVal
    sumAllLikelihood[1][DS] += likelihoodVal
    sumAllLikelihood[2][DS] += likelihoodVal
    sumAllLikelihood[3][0] += likelihoodVal
    sumAllLikelihood[4][0] += likelihoodVal
    sumAllLikelihood[4][1] += likelihoodVal


def runEM(trainData, whitenoiseCPTs):
    likelihood = None

    while True:
        sumJP = 0
        JPLLTable = newJPLLTable(whitenoiseCPTs)


        sumThisLikelihood = [[[0, 0, 0],\
                                 [0, 0, 0]],\
                                [0, 0, 0],\
                                [0, 0, 0],
                                [0],\
                                [0, 0]]

        sumAllLikelihood = [[[0, 0, 0],\
                             [0, 0, 0]],\
                            [0, 0, 0],\
                            [0, 0, 0],\
                            [0],\
                            [0, 0]]


        for data in trainData:
            if data[4] == -1:
                for DS in range(0, 3):
                    key = str([data[0], data[1], data[2], data[3], DS])
                    
                    sumJP += JPLLTable[key][0]
                    sumLikelihood(data, JPLLTable[key][1], sumAllLikelihood, sumThisLikelihood)
            else:
                sumJP += JPLLTable[str(data)][0]
                sumLikelihood(data, 1, sumAllLikelihood, sumThisLikelihood)


        # Updae S
        for THTS in range(0, 2):
            for DS in range(0, 3):
                whitenoiseCPTs[0][THTS][DS] = sumThisLikelihood[0][THTS][DS] / sumAllLikelihood[0][THTS][DS]
        
        # Update F D
        for DS in range(0, 3):
            whitenoiseCPTs[1][DS] = sumThisLikelihood[1][DS] / sumAllLikelihood[1][DS]
            whitenoiseCPTs[2][DS] = sumThisLikelihood[2][DS] / sumAllLikelihood[2][DS]

        # Update THTS
        whitenoiseCPTs[3][0] = sumThisLikelihood[3][0] / sumAllLikelihood[3][0]
        
        # Update DS
        for DS in range(0, 2):
            whitenoiseCPTs[4][DS] = sumThisLikelihood[4][DS] / sumAllLikelihood[4][0]
            whitenoiseCPTs[4][DS] = sumThisLikelihood[4][DS] / sumAllLikelihood[4][1]


        if (likelihood != None) and (sumJP - likelihood <= 0.01):
            break
        else:
            likelihood = sumJP


######################################################################################################################
######################################################################################################################
######################################################################################################################

def main():
    random.seed(None)

    # [[S [THTS [notDS Mild Severe]]
    #     [notTHTS [notDS Mild Severe]]]
    #  [F [notDS Mild Severe]]
    #  [D [notDS Mild Severe]]
    #  [THTS]
    #  [notDS Mild]]
    CPTs = [[[0.09, 0.32, 0.40],\
             [0.25, 0.05, 0.04]],\
            [0.12, 0.75, 0.26],\
            [0.28, 0.13, 0.85],\
            [0.1],\
            [0.5, 0.25]]
    
    trainData = []
    with open('traindata.txt', 'r') as f:
        for l in f:
            trainData.append([int(x) for x in l.strip().split()])

    testData = []
    with open('testdata.txt', 'r') as f:
        for l in f:
            testData.append([int(x) for x in l.strip().split()])
    

    meanAccuracyBefore = []
    meanAccuracyAfter = []

    stdDeviationBefore = []
    stdDeviationAfter = []

    delta = 0
    for i in range(0, 3):
        accuracyBefore = []
        accuracyAfter = []
        
        # should be 3 trails
        for j in range(0, 3):
            whitenoiseCPTs = addWhiteNoise(CPTs, delta)

            beforeEM = predict(testData, whitenoiseCPTs)
            accuracyBefore.append(beforeEM)

            runEM(trainData, whitenoiseCPTs)

            afterEM = predict(testData, whitenoiseCPTs)
            accuracyAfter.append(afterEM)
       

        mean = sum(accuracyBefore) / len(accuracyBefore)
        stdDeviation = np.std(accuracyBefore, ddof=1)

        meanAccuracyBefore.append(mean)
        stdDeviationBefore.append(stdDeviation)


        mean = sum(accuracyAfter) / len(accuracyAfter)
        stdDeviation = np.std(accuracyAfter, ddof=1)

        meanAccuracyAfter.append(mean)
        stdDeviationAfter.append(stdDeviation)

        delta += (4 / 3)


    print('Mean Accuracy\r')
    print('Delta  Before  After\r')
    for num in range(0, 3):
        print('{:<5.2f}  {:3.3f}  {:.3f}\r'.format(num * (4 / 3), meanAccuracyBefore[num], meanAccuracyAfter[num]))
                
    print('\r')

    print('Standard Deviation\r')
    print('Delta  Before  After\r')
    for num in range(0, 3):
        print('{:<5.2f}  {:3.3f}  {:.3f}\r'.format(num * (4 / 3), stdDeviationBefore[num], stdDeviationAfter[num]))
                
    print('\r')


main()
