# implement the baseline of LIME

import numpy as np
import json
import os
import string
import sys
from io import open
from rake_nltk import Rake
import keras
import pickle
from myutils import prep, drop, statusout, batch_gen, seq2sent, index2word, init_tf, seq2word, seqList2Sent
import progressbar
import time
from lime.lime_text import LimeTextExplainer
from Exp import Explain

# change the model file here
modelFile = ''
datstok = pickle.load(open('', 'rb'), encoding='UTF-8')
comstok = pickle.load(open('', 'rb'), encoding='UTF-8')
smltok = pickle.load(open('', 'rb'), encoding='utf-8')
model = keras.models.load_model(modelFile)

comlen = 13
comstart = np.zeros(comlen)
st = comstok.w2i['<s>']
comstart[0] = st

def translate(code, sbt=False, sml=None):
    words = code.split(' ')
    inpt = [np.zeros(100)]
    for i, w in enumerate(words):
        if i >= 100:
            break
        if w not in datstok.w2i.keys():
            inpt[0][i] = 0
        else:
            inpt[0][i] = datstok.w2i[w]
    coms = np.zeros(comlen)
    coms[0] = st
    coms = [coms]
        
    for i in range(1, comlen):
        if not sbt:
            results = model.predict([inpt, coms], batch_size=1)
        else:
            results = model.predict([inpt, coms, sml], batch_size=1)
        for c, s in enumerate(results):
            coms[c][i] = np.argmax(s)
    return seq2sent(coms[0], comstok).split(' ')

key = ''
batchSize = 200

def predictorLime(codeList):
    print(len(codeList))
    retList = []
    inpts = np.zeros((len(codeList), 100))
    coms = np.zeros((len(codeList), comlen))
    coms[:,0] = st

    for i, c in enumerate(codeList):
        for j, w in enumerate(c.split(' ')):
            if j >= 100:
                break
            if w not in datstok.w2i.keys():
                inpts[i][j] = 0
            else:
                inpts[i][j] = datstok.w2i[w]
    l = 0
    r = batchSize
    while l < len(codeList):
        tmpInpts = inpts[l: r]
        tmpComs = coms[l: r]
        for i in range(1, comlen):
            results = model.predict([tmpInpts, tmpComs], batch_size=r - l)
            for c, s in enumerate(results):
                coms[l + c][i] = np.argmax(s)
        r += batchSize
        l += batchSize
        if r >= len(codeList):
            r = len(codeList) - 1
    
    for c in coms:
        if key in seq2sent(c, comstok).split(' '):
            retList.append([0,1])
        else:
            retList.append([1,0])
    return np.array(retList)

def finalExplain_n(codes):
    resData = []
    r = Rake()
    classNames = ['negative', 'positive']
    exp = LimeTextExplainer(class_names=classNames)
    for j, code in enumerate(codes):
        tmpResult = {}
        c = translate(code)
        com = ''
        for i in range(1, len(c)):
            if c[i] == '</s>':
                break
            com += c[i] + ' '
        tmpResult['code'] = code
        tmpResult['comment'] = com
        r.extract_keywords_from_text(com)
        comKeys = r.get_ranked_phrases()
        tmpResult['commentKeywords'] = comKeys
            
        tmpList = []
        for _key in comKeys:
            global key
            key = _key
            tmpExp = {
                'commentKeyword': key,
            }
            explanation = exp.explain_instance(code, predictorLime, num_features=6)
            print(explanation.as_list())
            tmpExp['lime'] = explanation.as_list
            tmpList.append(tmpExp)
        tmpResult['explanations'] = tmpList
        resData.append(tmpResult)
    return resData

def finalExplain(codes):
    resData = []
    r = Rake()
    classNames = ['negative', 'positive']
    exp = LimeTextExplainer(class_names=classNames)
    with progressbar.ProgressBar(max_value=len(codes)) as bar:
        for j, code in enumerate(codes):
            tmpResult = {}
            c = translate(code)
            com = ''
            for i in range(1, len(c)):
                if c[i] == '</s>':
                    break
                com += c[i] + ' '
            tmpResult['code'] = code
            tmpResult['comment'] = com

            r.extract_keywords_from_text(com)
            comKeys = r.get_ranked_phrases()
            tmpResult['commentKeywords'] = comKeys
            
            tmpList = []
            for _key in comKeys:
                global key
                key = _key
                tmpExp = {
                    'commentKeyword': key,
                }
                explanation = exp.explain_instance(code, predictorLime, num_features=6, num_samples=5000)
                # print(explanation.as_list())
                tmpExp['lime'] = explanation.as_list()
                tmpList.append(tmpExp)
            tmpResult['explanations'] = tmpList
            resData.append(tmpResult)
            bar.update(j)
    return resData

def generateOutFile(inputFile, method, data):
    fname = inputFile + '_out_{}'.format(method)
    with open('outputs/' + fname, 'w') as f:
        json.dump(data, f, indent=2)
    return fname

def runTestFile(fileName, num, mode):
    timeRec = {}
    fname = fileName + '_{}'.format(str(num))
    pre =Explain('testFiles/' + fname, mode, model=model)
    codes = [s[0] for s in pre.tokenedCodes]
    st = time.time()
    result =finalExplain(codes)
    et = time.time()
    tmpName = generateOutFile(fname, 'lime', result)
    print(tmpName, et - st)
    with open('outputs/lime_{}_500_timer'.format(fileName), 'rw') as f:
        timeRec = json.load(f)
        timeRec[tmpName] = et - st
        json.dump(timeRec, f, indent=2)

def runTest(fileName, times, mode):
    timeRec = {}
    for i in range(times):
        fname = fileName + '_{}'.format(str(i))
        pre = Explain('testFiles/' + fname, mode, model=model)
        codes = [s[0] for s in pre.tokenedCodes]
        st = time.time()
        result = finalExplain(codes)
        et = time.time()
        tmpName = generateOutFile(fname, 'lime_5000', result)
        timeRec[tmpName] = et - st
    with open('outputs/lime_{}_5000_timer'.format(fileName), 'w') as f:
        json.dump(timeRec, f, indent=2)

if __name__ == '__main__':
    runTest('longCodeTestFile_DeepCom', 3, 1)
    runTest('longCodeTestFile_OriTest', 3, 101)
    runTest('longCodeTestFile_OriTrain', 3, 101)