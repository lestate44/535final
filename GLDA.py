from __future__ import division, print_function

import numpy as np
from gensim import corpora, models, similarities
import math
import pyLDAvis


def loadFile():
	f = open('src/5000h.txt', 'r')
	book = {}
	for line in f.readlines():
		# dataSet.append(line.strip())
		line = line.split()
		for i in range(3):
			line[i] = int(line[i])
		if line[0] not in book:
			book[line[0]] = {}
			book[line[0]][line[1]]=line[2]
		else:
			book[line[0]][line[1]]=line[2]
	bookList = []
	for key in sorted(book.keys()):
		bookList.append(book[key])
	book = [[0 for i in range(102660)] for i in range(len(bookList))]
	for i in range(len(bookList)):
		for key in bookList[i]:
			book[i][key-1] = bookList[i][key]
	new = []
	for i in range(len(book)):
		new.append([])
		for j in range(len(book[i])):
			if book[i][j] != 0:
				new[i].append((j, book[i][j]))
	return new

corpus=loadFile()

tfidf = models.TfidfModel(corpus)
lda = models.LdaModel(corpus,num_topics=5)
k=lda.bound(corpus,gamma=None)
print(k)