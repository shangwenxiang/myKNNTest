#coding=utf-8
from numpy import *
import operator
from os import listdir
import matplotlib
import matplotlib.pyplot as plt



def classify0(inX, dataSet, labels, k): #knn算法，距离计算，取前Ｋ个,进行排序
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


fileName='data/datingTestSet2.txt'	#数据加载
fr=open(fileName)
arrayLines=fr.readlines()	#读取全部数据
numberLines=len(arrayLines)	#获取数据长度	
MatData=zeros((numberLines,3))	#定义一个三维的数据
classLabelVector=[]
index=0
for line in arrayLines:
	line=line.strip()
	listFromLine=line.split('\t')
	MatData[index,:]=listFromLine[0:3]	#数据整理
	classLabelVector.append(int(listFromLine[-1]))	#标签整理
        index+=1

#数据显示
fig=plt.figure()
ax=fig.add_subplot(111)
ax.scatter(MatData[:,1],MatData[:,2])
ax.scatter(MatData[:,1],MatData[:,2],15.0*array(classLabelVector),15.0*array(classLabelVector))
#plt.show()  #取消注释可以可视化数据


minVals=MatData.min(0)	#最小数据
maxVals=MatData.max(0)	#最大数据
print 'minVals',minVals,'maxVals',maxVals
ranges=maxVals-minVals	#求取数据范围
normDataSet=MatData
m=MatData.shape[0]
normDataSet=MatData-tile(minVals,(m,1)) #数据归一化
normDataSet=normDataSet/tile(ranges,(m,1))
print normDataSet


hoRatio = 0.1
numTestVecs = int(m*hoRatio)	#取10％的数据进行数据测试
errorCount = 0.0
for i in range(numTestVecs):
    classifierResult = classify0(normDataSet[i,:],normDataSet[numTestVecs:m,:],classLabelVector[numTestVecs:m],3)
    if (classifierResult != classLabelVector[i]):	#结果计算
	errorCount += 1.0
print "the total error rate is: %f" % (errorCount/float(numTestVecs))
print errorCount	#最终错误结果
print double(errorCount/numTestVecs)	#错误结果百分率

