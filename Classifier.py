# coding=utf-8
import numpy as np
import math
import operator
import codecs


class KNN(object):
    def __init__(self, filename):
        self.filename = filename

    def file2matrix(self):
        with codecs.open(self.filename, 'r', encoding='utf-8') as fr:
            arrayOfLines = fr.readlines()
        numberOfLines = len(arrayOfLines)
        returnMat = np.zeros((numberOfLines, 320))

        classLabelVector = []
        index = 0
        for line in arrayOfLines:
            line = line.strip()
            listFromLine = line.split(' ')
            returnMat[index, :] = listFromLine[0:320]

            if listFromLine[-1] == '0':
                classLabelVector.append(0)
            elif listFromLine[-1] == '1':
                classLabelVector.append(1)

            index += 1
        return returnMat, classLabelVector

    # noinspection PyMethodMayBeStatic
    def classifyKNN(self, inputData, dataSet, labels, k):
        dataSetSize = dataSet.shape[0]

        diffMat = np.tile(inputData, (dataSetSize, 1)) - dataSet
        sqDiffMat = diffMat ** 2
        sqDistances = sqDiffMat.sum(axis=1)
        distances = sqDistances ** 0.5

        sortedDistIndices = distances.argsort()

        classCount = {}
        for i in range(k):
            voteLabel = labels[sortedDistIndices[i]]
            classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

            sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
            return sortedClassCount[0][0]

    def classifyVector(self, inputData):
        datingDataMat, datingLabels = self.file2matrix()
        classifierResult = self.classifyKNN(inputData, datingDataMat, datingLabels, 3)

        return classifierResult
