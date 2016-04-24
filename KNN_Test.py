
#coding=utf-8
from numpy import *
import operator


k=3  #定义k为3
arry=[0,0]
group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])#定义数据
labels = ['A','A','B','B']#定义数据标签
dataset=group.shape[0];#获取数据长度

diffmat=tile(arry,(dataset,1))-group #计算数据与arry的距离
sqdifmat=diffmat**2   
sqdis=sqdifmat.sum(axis=1) 
dis=sqdis**0.5

sorddis=dis.argsort()  #距离排序

classCount={}
for i in range(k):
	voteIlabel=labels[sorddis[i]] #标签对齐　给排序后的距离添加标签
	classCount[voteIlabel]=classCount.get(voteIlabel,0)+1#标签计数
sortedClassCount=sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True) #标签排序

print sortedClassCount[0][0]#输出最终结果





