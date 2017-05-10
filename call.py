import numpy as np
import lda
import lda.datasets
import math
import matplotlib.pyplot as plt
import operator


docfile='src/kos1000.txt'
vocfile='src/kosvoc.txt'

def loadv():
    f=open(vocfile,'r')
    vocab={}
    i=1
    for line in f.readlines():
        line=line.strip('\n')
        vocab[i]=line
        i=i+1
    return vocab

dict=loadv()


f = open(docfile, 'r')
file=open(str('src/ori.txt'),'w')
for line in f.readlines():
    line = line.split()
    st = int(line[1])
    if dict.has_key(st):
        file.write(line[0]+' '+dict[st]+'\n')
f.close()
file.close()
