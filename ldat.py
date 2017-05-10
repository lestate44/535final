import numpy as np
import lda
import lda.datasets
import math
import matplotlib.pyplot as plt
import operator


# preprocess and parameters set up
# define documents and vocabulary files:
docfile='src/kos1000.txt'
vocfile='src/kosvoc.txt'
newvoca= 'src/newvoc.txt'
logf="src/log.txt"

# get first document index
ff=open(docfile, 'r')
fline = ff.readline().split()
first = int(ff.readline().split()[0])
ff.close()

# define new temporary vocabulary dictionary:
f = open(docfile, 'r')
voc = []
for line in f.readlines():
    line = line.split()
    if line[1] not in voc:
        voc.append(line[1])
f.close()
voc.sort()
voclen = len(voc)
voca = {}
for i in range(voclen):
    voca[voc[i]] = i


# get total number of words
def gettotal():
    ld = open(docfile,'r')
    num=0
    for line in ld.readlines():
        line=line.split()
        num = num + int(line[2])
    return num

#create vocabulary dictionary:
def loadv():
    f=open(vocfile,'r')
    vocab={}
    i=1
    for line in f.readlines():
        line=line.strip('\n')
        vocab[i]=line
        i=i+1
    return vocab


f = open(docfile, 'r')
dict=loadv()
file=open(str('src/ori.txt'),'w')
for line in f.readlines():
    line = line.split()
    st = int(line[1])
    if dict.has_key(st):
        file.write(line[0]+' '+dict[st]+'\n')
f.close()
file.close()


# create its own vocabulary
def newvoc(name):

    ov=loadv()
    newv=[None]*voclen
    for i in range(voclen):
        for key in voca:
            if i==voca[key]:
                newv[i]=ov[int(key)]


    file=open(str(name),'w')
    for i in range(voclen):
        file.write(newv[i]+'\n')
    file.close()
    return name


# load data
def loaddata():
    f = open(docfile, 'r')
    book = {}
    for line in f.readlines():
        line = line.split()
        for i in range(3):
            line[i] = int(line[i])
        if line[0] not in book:
            book[line[0]] = {}
            book[line[0]][line[1]] = line[2]
        else:
            book[line[0]][line[1]] = line[2]
    mat = [[0 for i in range(voclen)] for i in range(len(book))]
    for i in range(len(book)):
        for key in book[i+first]:
            mat[i][voca[str(key)]] = book[i+first][key]
    return mat

# laod vocabulary
def loadVocab(name):
	f = open(str(name), 'r')
	vocab = []
	for line in f.readlines():
		line.strip('\n')
		vocab.append(line)
	return vocab

fw = open(logf, "w+")

# # train
data=loaddata()
vocab = loadVocab(newvoc(newvoca))
book = np.array(data)
model=lda.LDA(30,1000,1)
model.fit(book)
topic_word = model.topic_word_


#KL-divergence
def KL(i):
	a = np.asarray(topic_word[i], dtype=np.float64)
	temp = topic_word.tolist()
	temp.pop(i)
	b = [0 for i in range(len(topic_word[0]))]
	while temp:
		b = map(operator.add, b, temp.pop())
	b = np.asarray(b, dtype=np.float64)
	return np.sum(np.where(a != 0, a * np.log(a / b), 0))
klDiv = []
for i in range(0, len(topic_word)-1):
	klDiv.append(KL(i))

# topic top words display
n = 10
for i, topic_dist in enumerate(topic_word):
	topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n+1):-1]
	temp = np.argsort(topic_dist)[::-1]
	mSum = float(np.argsort(topic_dist)[:-21:-1].sum())
	p = [t/mSum for t in temp]
	fw.write('*Topic {}\n {}\n'.format(i, ' '.join(str(j) for j in zip(topic_words, p))))

# document-topics plot
#
doc_topic = model.doc_topic_
for n in range(len(book)):
	topic_most_pr = doc_topic[n].argmax()
	fw.write("doc: {} topic: {}\n".format(n, topic_most_pr))

for i in range(len(klDiv)):
	fw.write('KL-divergence of Document {}: {}\n'.format(i+1, klDiv[i]))

f,ax=plt.subplots(6,1,figsize=(10,10),sharex=True)
for i,k in enumerate([0,1,2,5,7,8]):
    ax[i].stem(doc_topic[k,:],linefmt='r-',marketfmt='ro',basefmt='w-')
    ax[i].set_xlim(-1,21)
    ax[i].set_ylim(0,1.2)
    ax[i].set_ylabel("Prob")
    ax[i].set_title("Document {}".format(k))
ax[5].set_xlabel("Topic")
plt.tight_layout()

# perplexity vs topics plot:
num=gettotal()
ntop=[]
per=[]
for top in range(2,51):
	model=lda.LDA(top,600,1)
	model.fit(book)
	ntop.append(top)
	k=model.loglikelihood()
	ll=float(k)
	ll=math.exp(-k/num)
	per.append(ll)
plt.figure(2)
plt.plot(ntop,per)
plt.xlabel('Number of Topics')
plt.ylabel('Perplexity')
plt.show()