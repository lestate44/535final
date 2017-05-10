import plsa
import matplotlib.pyplot as plt
import math
import numpy as np

docFile = "src/1000.txt"
vocabFile = 'src/vocab.txt'
logFile = "src/plsaLog.txt"

# get total number of words in corpus:
def gettotal():
    ld = open(docFile,'r')
    num=0
    for line in ld.readlines():
        line=line.split()
        num = num + int(line[2])
    return num

#convert file
def loadfile():
    af = open(docFile, 'r')
    book={}
    first = af.readline().split()
    k = af.readline().split()[0]
    word = af.readline().split()[2]
    af.close()
    book = {}
    booklist=[]
    f = open(docFile, 'r')
    for line in f.readlines():
        line = line.split()
        for i in range(3):
            line[i] = int(line[i])
        if(line[0]==int(k)):
            book[line[1]]=int(line[2])
        else:
            booklist.append(book)
            k=line[0]
            book={}
            book[line[1]]=int(line[2])

    return booklist
print(len(loadfile()[0]))

#Load dictionary
def loadVocab():
    f = open(vocabFile, 'r')
    vocab = []
    for line in f.readlines():
        line.strip('\n')
        vocab.append(line)
    return vocab

def showTopic(seq):
    vocab = loadVocab()
    for i in range(20):
        temp = sorted(seq[i])
        fw.write('Topic {}:\n'.format(i))
        for j in range(10):
            fw.write('{}'.format(vocab[seq[i].index(temp[-(j+1)])-1]))

def docTopic(tpc):
    for  i in range(len(tpc)):    # Num of docs to show
        temp = sorted(tpc[i])
        idx = tpc[i].index(temp[-1])
        fw.write('Document {}: Topic {} \n'.format(i, idx))

# train and plot
fw = open(logFile, "w+") #logFile

ind = gettotal()
like=[]
topics=[]

print(ind)
#train model
for topic in range(2,21): #Set topics
    k=plsa.Plsa(loadfile(),topic)
    k.train(5) #Set iteration
    ll=float(k.likelihood)
    ll=math.exp(-ll/ind)
    like.append(ll)
    topics.append(int(topic))

#Log topics
showTopic(k.zw)
docTopic(k.dz)

#Output document-topic distribution
docTpc = np.array(k.dz)
f,ax=plt.subplots(6,1,figsize=(10,10),sharex=True)
for i,k in enumerate([0,1,2,5,7,8]):
    ax[i].stem(docTpc[k,:],linefmt='r-',marketfmt='ro',basefmt='w-')
    ax[i].set_xlim(-1,20)
    ax[i].set_ylim(0,1.2)
    ax[i].set_ylabel("Prob")
    ax[i].set_title("Document {}".format(k))
ax[5].set_xlabel("Topic")
plt.tight_layout()


#
#
#
#Output perplexity
ind = gettotal()
like=[]
topics=[]
for topic in range(2,50): #Set topics
    k=plsa.Plsa(loadfile(),topic)
    k.train(5) #Set iteration
    ll=float(k.likelihood)
    ll=math.exp(-ll/ind)
    like.append(ll)
    topics.append(int(topic))
print(like)
print(topics)

plt.figure(2)
plt.plot(topics,like)
plt.xlabel('Number of Topics')
plt.ylabel('Perplexity')
plt.show()




# k=plsa.Plsa(loadfile(),2)
# k.train(10)
# ll=k.likelihood
# print ll


