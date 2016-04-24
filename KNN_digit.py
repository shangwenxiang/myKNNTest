#coding=utf-8
from numpy import *
import operator
from os import listdir


def classify0(inX, dataSet, labels, k):#knn算法
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()     
    classCount={}          
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def img2vector(filename):	#加载文件，把文件变为1024维度的数组，文件为32*#2
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():	#手写数字测试
    hwLabels = []
    trainingFileList = listdir('data/trainingDigits')#训练文件列表加载
    m = len(trainingFileList)	#获取文件列表的长度
    trainingMat = zeros((m,1024))	#定义1024*m的数组
    for i in range(m):			#文件加载到数组中
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('data/trainingDigits/%s' % fileNameStr)
    testFileList = listdir('data/testDigits')	#测试文件列表加载
    errorCount = 0.0
    mTest = len(testFileList)	#获取测试文件列表长度
    for i in range(mTest):	#将文件加载到数组中并测试
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('data/testDigits/%s' % fileNameStr)#加载完毕
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))

handwritingClassTest()
