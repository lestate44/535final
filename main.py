import collections
import numpy as np
import lda
import math
import matplotlib.pyplot as plt

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
	return book

def loadVocab():
	f = open('src/vocab.txt', 'r')
	vocab = []
	for line in f.readlines():
		line.strip('\n')
		vocab.append(line)
	return vocab
	# vocab.append(line for line in f.readlines())
print len(loadFile()[0])


# LDA train

book = loadFile()
vocab = loadVocab()
book = np.array(book)

# ntop=[]
# per=[]
# for top in range(2,50):
# 	model=lda.LDA(top,100,1)
# 	model.fit(book)
# 	ntop.append(top)
# 	k=model.loglikelihood()
# 	ll=float(k)
# 	ll=math.exp(-k/117286)
# 	per.append(ll)
# plt.plot(ntop,per)
# plt.xlabel('Number of Topics')
# plt.ylabel('Perplexity')
# plt.show()


model = lda.LDA(n_topics=2, n_iter=100, random_state=1)
model.fit(book)
ll=model.loglikelihood()
print(ll)
ll=math.exp(-ll/32581)
print(ll)


# topic top words display
#
topic_word = model.topic_word_
n = 5
for i, topic_dist in enumerate(topic_word):
	topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n+1):-1]
	temp = np.argsort(topic_dist)[::-1]
	mSum = float(np.argsort(topic_dist)[:-21:-1].sum())
	p = [t/mSum for t in temp]
	print('*Topic {}\n {}'.format(i, ' '.join(str(j) for j in zip(topic_words, p))))


# document-topics plot
#
# doc_topic = model.doc_topic_
# import matplotlib.pyplot as plt
# f,ax=plt.subplots(6,1,figsize=(10,10),sharex=True)
# for i,k in enumerate([0,1,2,5,7,8]):
#     ax[i].stem(doc_topic[k,:],linefmt='r-',marketfmt='ro',basefmt='w-')
#     ax[i].set_xlim(-1,10)
#     ax[i].set_ylim(0,1.2)
#     ax[i].set_ylabel("Prob")
#     ax[i].set_title("Document {}".format(k))
# ax[5].set_xlabel("Topic")
# plt.tight_layout()
# plt.show()



# topic-words plot
# f, ax = plt.subplots(2, 1, figsize=(20, 6), sharex=True)
# for i, k in enumerate([0, 1]):
# 	ax[i].stem(topic_word[k, :], linefmt='b-',
# 			   markerfmt='bo', basefmt='w-')
# 	ax[i].set_xlim(-2, 103000)
# 	ax[i].set_ylim(0, 1)
# 	ax[i].set_ylabel("Prob")
# 	ax[i].set_title("topic {}".format(k))
#
# ax[1].set_xlabel("word")
#
# plt.tight_layout()
# plt.show()
# for n in range(len(book)):
# 	topic_most_pr = doc_topic[n].argmax()
# 	print("doc: {} topic: {}".format(n, topic_most_pr))
# doc_topic = model.doc_topic_
# titles = lda.datasets.load_reuters_titles()
# print(titles)
# for i in range(10):
#    print("{} (top topic: {})".format(book[i], doc_topic[i].argmax()))