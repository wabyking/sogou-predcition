#coding=utf-8
from __future__ import division
import os
import random,math
import cPickle as pickle
import pandas as pd
import ConfigParser
from sklearn.linear_model import LinearRegression 
import codecs
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer
from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn import cross_validation

from sklearn.ensemble import RandomForestClassifier
import requests
import jieba


conf = ConfigParser.ConfigParser()   
conf.read("arg.cfg") 
stopwords_path=conf.get("path","stopwords_path")
stopwords={word.decode("utf-8") for word in open(stopwords_path).read().split()}

def cut(sentence):

	sentence=" ".join([item for item in sentence.split("\t") if "http" not in item ] )
	# try:
	# 	words= pynlpir.segment(sentence, pos_tagging=False)
	# except:
	try:
		words= jieba.cut(sentence)
	except:
		words=[]
	
	words=[word for word in words if word not in stopwords]
	# print words
	return words

def getTest():
	
	test_path=conf.get("path","test_path")
	records=[]
	with codecs.open(test_path,"r","GB18030") as f:
		for i,line in enumerate(f):
			tokens=line.split("\t")
			ID = tokens[0]
			terms=tokens[1:]
			record={"ID":ID,"terms":"\t".join(terms)}
			records.append(record)
	return pd.DataFrame(records)

def getTrain():
	
	train_path=conf.get("path","train_path")
	records=[]
	with codecs.open(train_path,"r","GB18030") as f:
		for i,line in enumerate(f):
			tokens=line.split("\t")
			ID = tokens[0]
			age= int(tokens[1])
			gender=int(tokens[2])
			education=int(tokens[3])
			terms=tokens[4:]
			record={"ID":ID,"age":age,"gender":gender,"education":education,"terms":"\t".join(terms)}
			records.append(record)
	return pd.DataFrame(records)
			# print record

def loadData(option="offline" ,fresh=False,demo=False):
	if option=="offline":
		clearedpath=conf.get("temp","offline")
		if os.path.exists(clearedpath) and not fresh:
			train,test=pickle.load(open(clearedpath,'r'))
			return train,test
		train,test=getTrainAndTest()
		pickle.dump((train,test),open(clearedpath,"w"))
		if demo:
			return train[:1000],test[:1000]	
		return train,test
	else:
		clearedpath=conf.get("temp","online")
		if os.path.exists(clearedpath):
			train,test=pickle.load(open(clearedpath,'r'))
			return train,test
		test=getTest()
		train=getTrain()
		text_clear=lambda row: " ".join(cut(row["terms"]))
		train["clearedtext"]=train.apply(text_clear,axis=1)
		test["clearedtext"]=test.apply(text_clear,axis=1)
		pickle.dump((train,test),open(clearedpath,"w"))	
		return train,test


def getTrainAndTest(rate=0.7):
	
	df=getTrain()
	text_clear=lambda row: " ".join(cut(row["terms"]))
	df["clearedtext"]=df.apply(text_clear,axis=1)

	size=len(df)
	flags=[True] * int(size*rate) + [False] *  (size-int(size*rate))
	random.seed(822)
	random.shuffle(flags)
	train=df[flags]
	test=df[map(lambda x:not x, flags)]
	return train,test

def main():
	train,test=loadData(option="offline",fresh=False,demo=False)  
	fields=["age","gender","education"]

	# enough={"age":[1,2,3,4],"gender":[1,2],"education":[3,4,5]}
	if len(test)>=20000:
		for field in fields:
			if field=="gender":
				train=train[train.gender!=0]
			test[field+"_p"]=predict(train, test,field)
		names=[field+"_p" for field in fields]
		test[(["ID"]+names)].to_csv("submission.csv",header=False,index=False,encoding="gbk",sep=' ')
	else:
		predicts=[]
		for field in fields:
			# train=train[train[field].isin(enough[field])]
			train=train[train[field]!=0]
			predicts.append(predict(train, test,field))
		print predicts,sum(predicts)/3
		
		

def predict(train,test, field="gender"):
	
	nbc = Pipeline([
		    ('vect', TfidfVectorizer()),
		    ('clf', BernoulliNB(alpha=0.3) ),   #BernoulliNB(alpha=0.3) ,max_features="sqrt"
		])
	nbc.fit(train["clearedtext"], train[field])    #训练我们的多项式模型贝叶斯分类器
	predicted = nbc.predict(test["clearedtext"])
	if len(test)>=20000:
		return predicted
	count=0
	for left , right in zip(predicted, test[field]):
	      if left == right:
	            count += 1
	return count/len(predicted)
	
if __name__ == '__main__':
	main()
