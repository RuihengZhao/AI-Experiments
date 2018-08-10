import numpy as np
from functools import reduce

class Factor(object):
    def __init__(self, vars, p):
        self.vars = vars
        self.probabilityTable = np.array(p)

    def printTable(self):
        print(self.vars)
        print(self.probabilityTable)
        print('\r')

    @staticmethod
    def restrict(factor, variable, value):
        index = factor.vars.index(variable)
        del factor.vars[index]

        sliceTuple = ()
        for i in range(0, index):
            sliceTuple += slice(None),

        if value == 't':
            sliceTuple += slice(0, 1),
        else:
            sliceTuple += slice(1, 2),
        
        factor.probabilityTable = factor.probabilityTable[sliceTuple]

        if len(factor.vars) != 0:
            if len(factor.probabilityTable[0]) == 1:
                factor.probabilityTable = np.array([factor.probabilityTable[0][0], factor.probabilityTable[1][0]])
            else:
                factor.probabilityTable = factor.probabilityTable[0]
        else:
            factor.probabilityTable = factor.probabilityTable[0]

        return factor

    @staticmethod
    def multiply(factor1, factor2):
        newVars = list(factor1.vars)
        for var in factor2.vars:
            if not (var in newVars):
                newVars.append(var)
                
        newVars.sort()

        factor = Factor([],[])
        factor.vars = newVars
        factor.probabilityTable = factor1.probabilityTable * factor2.probabilityTable

        return factor

    @staticmethod
    def sumout(factor, variable):
        index = factor.vars.index(variable)
        del factor.vars[index]
        
        factor.probabilityTable = factor.probabilityTable.sum(axis = index)
        
        return factor

    @staticmethod
    def normalize(factor):
        factor.probabilityTable = factor.probabilityTable / factor.probabilityTable.sum()

        return factor
    
    @staticmethod
    # resultFactor = inference(factorList, queryVariables, orderedListOfHiddenVariables, evidenceList)
    def inference(factorList, queryVariables, orderedListOfHiddenVariables, evidenceList):
        for e in evidenceList:
            restrictedFactorList = []
            for factor in factorList:
                if (e in factor.vars):
                    print('Restrict:', e)

                    restrictedFactor = Factor.restrict(factor, e, evidenceList[e])
                    restrictedFactor.printTable()

                    restrictedFactorList.append(restrictedFactor)
                else:
                    restrictedFactorList.append(factor)

            factorList = restrictedFactorList

        for var in orderedListOfHiddenVariables:
            multiplyFactors = []
            otherFactors = []

            for factor in factorList:
                if var in factor.vars:
                    multiplyFactors.append(factor)
                else:
                    otherFactors.append(factor)
            
            print('Multiply:', var)
            multipliedFactors = multiplyFactors[0]
            for i in range(1, len(multiplyFactors)):
                multipliedFactors = Factor.multiply(multipliedFactors, multiplyFactors[i])
            multipliedFactors.printTable()
            
            print('Sumout:',var)
            sumoutFactors = Factor.sumout(multipliedFactors, var)
            sumoutFactors.printTable()
            
            otherFactors.append(sumoutFactors)
            factorList = otherFactors

        print('Multiply other factors:')
        multipliedFactors = factorList[0]
        for i in range(1, len(factorList)):
            multipliedFactors = Factor.multiply(multipliedFactors, factorList[i])
        multipliedFactors.printTable()

        print('Normalize')
        normalizedFactors = Factor.normalize(multipliedFactors)
        normalizedFactors.printTable()

        return normalizedFactors
