# the experiment mehtod proposed in the paper

import os
import sys
import traceback
import pickle
import argparse
import collections
import random

from keras import metrics
import tensorflow as tf
import numpy as np

seed = 1337
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

from rake_nltk import Rake
from apriori import apriori
import json

from myutils import seq2sent, seq2word, seqList2Sent
import keras

from myTokenizer import Token, Keyword
import progressbar
import time

# change the model file here
modelFile = ''
sbtModelFile = ''

# change the token file for dataset here
datstokFile = ''
comstokFile = ''
smltokFile = ''


comlen = 13
comstart = np.zeros(comlen)

class Explain():
    def __init__(self, _codeFile, _mode, model=None, step=0.05, numSamples=100):
        super().__init__()
        self.datstok = pickle.load(open(datstokFile, 'rb'), encoding='UTF-8')
        self.comstok = pickle.load(open(comstokFile, 'rb'), encoding='UTF-8')
        self.smltok = pickle.load(open(smltokFile, 'rb'), encoding='utf-8')
        self.model = model
        if model == None:
            self.model = self.loadModel(_mode)
        if _codeFile != None:
            self.rawData, self.num = self.loadData(_codeFile, _mode)
            self.tokenedCodes = self.tokenizeCodes(self.rawData, _mode)
            if not self.checkMode(_mode, 'withSbt'):
                self.sbts = [None] * len(self.tokenedCodes)
        self.mode = _mode
        self.sml = None
        self.r = Rake()
        self.tokenizer = Token()
        self.st = self.comstok.w2i['<s>']
        self.step = step
        self.numSamples = numSamples
    
    def reload(self, _codeFile, _mode):
        self.rawData, self.num = self.loadData(_codeFile, _mode)
        self.tokenedCodes = self.tokenizeCodes(self.rawData, _mode)
    
    def reloadData(self, _codeFile, _mode):
        self.rawData, self.num = self.loadData(_codeFile, _mode)
        self.tokenedCodes = self.tokenizeCodes(self.rawData, _mode)

    def explain_n(self):
        print('start')
        retData = []
        with progressbar.ProgressBar(max_value=self.num) as bar:
            for m in range(self.num):
                tmpData = {}
                code = self.tokenedCodes[m]
                # print(self.sbts[m])
                com, c = self.translateStrs(code, self.checkMode(self.mode, 'withSbt'), self.sbts[m])
                tmpData['code'] = code
                tmpData['comment'] = com

                self.r.extract_keywords_from_text(com)
                comKeys = self.r.get_ranked_phrases()
                tmpData['commentKeywords'] = comKeys

                codeWordList = self.tokenizer.toDoubleList(code)
                codeKeys, codeKeyIndex = self.extractCodeKeys(code)
                tmpData['codeKeywords'] = codeKeys
                tmpData['codeKeyIndex'] = codeKeyIndex
                tmpList = []
                retSamples = self.explainMultiKey(codeWordList, self.sbts[m], codeKeyIndex, self.numSamples, comKeys, 0.6)
                for index, sample in retSamples.items():
                    tmpResults = {
                        'commentKeyword': comKeys[index],
                    }
                    L, support = apriori(sample, 0.3)
                    L = [[[int(j) for j in i] for i in l] for l in L]
                    support = [[[int(i) for i in s[0]], s[1]] for s in support.items()]
                    tmpResults['anchors'] = L
                    tmpResults['supports'] = support
                    tmpList.append(tmpResults)
                tmpData['explanations'] = tmpList
                retData.append(tmpData)
                bar.update(m)
        return retData
    
    def explain(self):
        print('start')
        retData = []
        with progressbar.ProgressBar(max_value=self.num) as bar:
            for m in range(self.num):
                tmpData = {}
                code = self.tokenedCodes[m]
                # print(self.sbts[m])
                com, c = self.translateStrs(code, self.checkMode(self.mode, 'withSbt'), self.sbts[m])
                tmpData['code'] = code
                tmpData['comment'] = com

                self.r.extract_keywords_from_text(com)
                comKeys = self.r.get_ranked_phrases()
                tmpData['commentKeywords'] = comKeys

                codeWordList = self.tokenizer.toDoubleList(code)
                codeKeys, codeKeyIndex = self.extractCodeKeys(code)
                # codeKeys, codeKeyIndex = self.extractCodeKeysn(code)
                tmpData['codeKeywords'] = codeKeys
                tmpData['codeKeyIndex'] = codeKeyIndex
                tmpList = []
                for key in comKeys:
                    tmpResults = {
                        'commentKeyword': key,
                    }
                    keyNums = np.zeros(len(codeKeyIndex))
                    i, ni, p = self.explainKey(codeWordList, self.sbts[m], codeKeyIndex, self.numSamples, key, 0.6)
                    tmpResults['numberHaveKey'] = len(i)
                    tmpResults['numberNoKey'] = len(ni)
                    tmpResults['probability'] = p
                    for keyIds in ni:
                        tmp = list(set(keyIds))
                        for id in tmp:
                            keyNums[id] += 1
                    L, support = apriori(ni, 0.3)
                    L = [[[int(j) for j in i] for i in l] for l in L]
                    support = [[[int(i) for i in s[0]], s[1]] for s in support.items()]
                    tmpResults['anchors'] = L
                    tmpResults['supports'] = support
                    tmpList.append(tmpResults)
                tmpData['explanations'] = tmpList
                retData.append(tmpData)
                bar.update(m)
        return retData
    
    @staticmethod
    def ndarray2List(ndarray):
        retList = [int(i) for i in ndarray]
        return retList

    def explainLine(self):
        retData = []
        with progressbar.ProgressBar(max_value=self.num) as bar:
            for m in range(self.num):
                tmpData = {}
                code = self.tokenedCodes[m]
                com, c = self.translateStrs(code, self.checkMode(self.mode, 'withSbt'), self.sbts[m])
                tmpData['code'] = code
                tmpData['comment'] = com

                self.r.extract_keywords_from_text(com)
                comKeys = self.r.get_ranked_phrases()
                tmpData['commentKeywords'] = comKeys

                tmpList = []
                for key in comKeys:
                    tmpResults = {
                        'commentKeyword': key,
                    }
                    lineNum = np.zeros(len(code))
                    i, ni = self.explainLineBase(code, key)
                    tmpResults['numberHaveKey'] = i
                    tmpResults['numberNoKey'] = ni
                    for i, j in ni:
                        lineNum[i: j] += 1
                    tmpResults['lineWeight'] = self.ndarray2List(lineNum)
                    tmpList.append(tmpResults)
                tmpData['explanations'] = tmpList
                retData.append(tmpData)
                bar.update(m)
        return retData

    def explainLineBase(self, code, key):
        inComs = []
        notInComs = []

        for i in range(len(code)):
            for j in range(0, len(code) - i):
                temp = code[j: j + i + 1]
                blockLines = []
                for l in temp:
                    words = l.split(' ')
                    t = ['<UNK>' for w in words]
                    t = ' '.join(t)
                    blockLines.append(t)
                tempCode = code.copy()
                tempCode[j: j + i + 1] = blockLines
                com, _ = self.translateStrs(tempCode, self.checkMode(self.mode, 'withSbt'), self.sml)
                if key in com:
                    inComs.append((j, j + i + 1))
                else:
                    notInComs.append((j, j + i + 1))
        return inComs, notInComs

    def explainKey(self, codeWordList, _sml, codeKeyIndex, numSamples, comKey, thresh=0.6):
        p = 0.1
        step = self.step
        while p <= 1.0:
            i, ni = self.explainKeyOnce(codeWordList, _sml, codeKeyIndex, numSamples, comKey, p, thresh)
            if len(ni) >= numSamples:
                break
            else:
                p += step
        return i, ni, p

    def explainMultiKey(self, codeWordList, _sml, codeKeyIndex, numSamples, comKeyList, thresh=0.6):
        p = 0.1
        step = self.step
        tmpKeyList = comKeyList.copy()
        retSample = {}
        deleted = [0] * len(tmpKeyList)
        while p <= 1.0:
            iList, niList = self.explainKeyOnce_n(codeWordList, _sml, codeKeyIndex, numSamples, tmpKeyList, deleted, p, thresh)
            assert len(niList) == len(comKeyList)
            for i, l in enumerate(niList):
                if len(l) >= numSamples and not deleted[i]:
                    retSample[i] = l
                    deleted[i] = 1
            if len(tmpKeyList) == sum(deleted):
                break
            else:
                p += step
        return retSample

    
    def explainKeyOnce_n(self, code, _sml, key, numsamples, comKeyList, deleted, p, thresh):
        codes = []
        keyIds = []
        inComs = []
        notInComs = []
        for i in range(int(numsamples / thresh)):
            numChanged = np.random.binomial(len(key), min(p, 1.0))
            changed = np.random.choice(len(key), numChanged)
            keyIds.append(changed)
            codes.append(self.generateCode(code, key, changed))
        coms = self.translateBatch(codes, _sml)
        for j, comKey in enumerate(comKeyList):
            tmpIn = []
            tmpNotIn = []
            add = 0
            if deleted[j]:
                inComs.append(tmpIn)
                notInComs.append(tmpNotIn)
                continue
            for i, com in enumerate(coms):
                if comKey in com:
                    tmpIn.append(keyIds[i])
                else:
                    add += 1
                    tmpNotIn.append(keyIds[i])
            inComs.append(tmpIn)
            notInComs.append(tmpNotIn)
        return inComs, notInComs

    def explainKeyOnce(self, code, _sml, key, numSamples, comKey, p, thresh):
        codes = []
        keyIds = []
        inComs = []
        notInComs = []
        for i in range(int(numSamples / thresh)):
            numChanged = np.random.binomial(len(key), min(p, 1.0))
            changed = np.random.choice(len(key), numChanged)
            keyIds.append(changed)
            codes.append(self.generateCode(code, key, changed))
        coms = self.translateBatch(codes, _sml)
        for i, com in enumerate(coms):
            if comKey in com:
                inComs.append(keyIds[i])
            else:
                notInComs.append(keyIds[i])
        return inComs, notInComs
    
    @staticmethod
    def generateCode(_code, key, ids):
        code = [i.copy() for i in _code]
        for id in ids:
            codeKey = key[id]
            lineNum = codeKey[0]
            words = codeKey[1]
            for i in words:
                code[lineNum][i] = '<unk>'
        retCode = ''
        for l in code:
            retCode += ' '.join(l) + ' '
        return retCode
    
    def extractCodeKeysn(self, code):
        codeKeys = []
        codeKeyIndex = []
        k = Keyword()
        for i, l in enumerate(code):
            tmpKeys = [j for j in l.split(' ')]
            t = [(i, [j]) for j in range(len(tmpKeys))]
            codeKeyIndex += t
            codeKeys += tmpKeys
        return codeKeys, codeKeyIndex

    def extractCodeKeys(self, code):
        codeKeys = []
        codeKeyIndex = []
        k = Keyword()
        for i, l in enumerate(code):
            self.r.extract_keywords_from_text(l)
            tmpKeys = self.r.get_ranked_phrases()
            t, filteredKey = k.preprocess(tmpKeys, i, l)
            codeKeyIndex += t
            codeKeys += filteredKey
        return codeKeys, codeKeyIndex

    def translateBatch(self, codeList, _sml):
        inp = np.zeros((len(codeList), 100))
        for i, code in enumerate(codeList):
            for j, w in enumerate(code.split(' ')):
                if j == 100:
                    break
                if w not in self.datstok.w2i.keys():
                    inp[i][j] = 0
                else:
                    inp[i][j] = self.datstok.w2i[w]
        coms = np.zeros((len(codeList), comlen))
        coms[:, 0] = self.st
        
        for i in range(1, comlen):
            if not self.checkMode(self.mode, 'withSbt'):
                results = self.model.predict([inp, coms], batch_size=len(codeList))
            else:
                sml = np.zeros((len(codeList), 100))
                sml[:, :] = np.array(_sml)
                results = self.model.predict([inp, coms, sml], batch_size=len(codeList))
            for c, s in enumerate(results):
                coms[c][i] = np.argmax(s)
        strComs = [i.split(' ') for i in seqList2Sent(coms, self.comstok)]
        transferedComs = []
        for com in strComs:
            s = ''
            for j in range(1, len(com)):
                if com[j] == '</s>':
                    break
                s += com[j] + ' '
            transferedComs.append(s)
        return [i.split(' ')[: -1] for i in transferedComs]

    def translateStrs(self, code, sbt=False, sml=None):
        com = ''
        inp = ' '.join(code)
        c = self.translate(inp, sbt, sml)
        for i in range(1, len(c)):
            if c[i] == '</s>':
                break
            com += c[i] + ' '
        return com, c

    def translate(self, code, sbt, sml):
        words = code.split(' ')
        inp = [np.zeros(100)]
        for i, w in enumerate(words):
            if i >= 100:
                break
            if w not in self.datstok.w2i.keys():
                inp[0][i] = 0
            else:
                inp[0][i] = self.datstok.w2i[w]
        coms = np.zeros(comlen)
        coms[0] = self.st
        coms = [coms]

        for i in range(1, comlen):
            if not sbt:
                results = self.model.predict([inp, coms], batch_size=1)
            else:
                # print(sml)
                results = self.model.predict([inp, coms, [sml]], batch_size=1)
            for c, s in enumerate(results):
                coms[c][i] = np.argmax(s)
        return seq2sent(coms[0], self.comstok).split(' ')

    def tokenizeCodes(self, _rawCodes, _mode):
        tokenizer = Token()
        if self.checkMode(_mode, 'withToken'):
            tokenedCodes = _rawCodes
            if self.checkMode(_mode, 'needLine'):
                tokenedCodes = tokenizer.splitLinesWithFixedLength(_rawCodes)
        else:
            linedCodes = _rawCodes
            if self.checkMode(_mode, 'needLine'):
                linedCodes = tokenizer.generateLinesFromRawCodes(_rawCodes)
            tokenedCodes = tokenizer.getFromLines(linedCodes)
        return tokenedCodes
            
    def loadData(self, _codeFile, _mode):
        # load raw data from different sources
        if self.checkMode(_mode, 'fromFile'):
            code, num = self.loadFromFile(_codeFile, _mode)
        else:
            code, num = self.loadFromString(_codeFile, _mode)
        return code, num

    def loadFromFile(self, _codeFile, _mode):
        if self.checkMode(_mode, 'withLine'):
            code, num = self.loadFromLinedFile(_codeFile, _mode)
        else:
            # 无分行的文件为数据集
            code, num = self.loadDataset(_codeFile, _mode)
        return code, num

    def loadSbt(self, _sbtFile):
        sbt = []
        with open(_sbtFile) as f:
            for line in f.readlines():
                indexes = [int(i) for i in line.split(' ')]
                sbt.append(np.array(indexes))
        self.sbts = np.array(sbt)

    def loadDataset(self, _codeFile, _mode):
        codes = []
        sbts = []
        with open(_codeFile) as f:
            for line in f.readlines():
                codes.append([line])
        if self.checkMode(_mode, 'withSbt'):
            fname = _codeFile + '_sbt'
            self.loadSbt(fname)
        return codes, len(codes)

    def loadFromString(self, _codeStr, _mode):
        if self.checkMode(_mode, 'withLine'):
            code = _codeStr.split('\n')
        else:
            code = _codeStr
        return [code], 1
            
    def loadFromLinedFile(self, _codeFile, _mode):
        code = []
        with open(_codeFile) as f:
            for line in f.readlines():
                code.append(line)
        return [code], 1

    def loadModel(self, _mode):
        keras.backend.clear_session()
        if self.checkMode(_mode, 'withSbt'):
            model = keras.models.load_model(sbtModelFile)
        else:
            model = keras.models.load_model(modelFile)
        return model

    @staticmethod
    def genMode(_withLines, _withToken, _withSbt, _fromFile, _needLine):
        mode = 0
        if _needLine:
            mode = mode * 10 + 1
        if _withLines:
            mode = mode * 10 + 1
        if _withToken:
            mode = mode * 10 + 1
        if _withSbt:
            mode = mode * 10 + 1
        if _fromFile:
            mode = mode * 10 + 1
        return mode
    
    @staticmethod
    def checkMode(_mode, _str):
        if _str == 'needLine':
            return (_mode // 10000) % 10 == 1
        if _str == 'withLine':
            return (_mode // 1000) % 10 == 1
        if _str == 'withToken':
            return (_mode // 100) % 10 == 1
        if _str == 'withSbt':
            return (_mode // 10) % 10 == 1
        if _str == 'fromFile':
            return (_mode) % 10 == 1

def generateOutFile(inputFile, method, data, mode):
    fname = inputFile + '_out_{}'.format(method)
    if mode == 111:
        fname = fname + '_sbt'
    with open('outputs/' + fname, 'w') as f:
        json.dump(data, f, indent=2)
    return fname

def runTestKey(FileName, times, step, numSamples, mode):
    # keras.backend.clear_session()
    timeRec = {}
    for i in range(times):
        fname = FileName + '_{}'.format(str(i))
        exp = Explain('testFiles/' + fname, mode, model=None, step=step, numSamples=numSamples)
        st = time.time()
        resKeyword = exp.explain()
        et = time.time()
        tmpName = generateOutFile(fname, 'keyword_step{}_{}'.format(step, numSamples), resKeyword, mode)
        timeRec[tmpName] = et - st
    if mode == 111:
        with open('outputs/keywordstep{}_{}_sbt_timer'.format(str(step), FileName), 'w') as f:
            json.dump(timeRec, f, indent=2)
    else:
        with open('outputs/keywordstep{}_{}_timer'.format(str(step), FileName), 'w') as f:
            json.dump(timeRec, f, indent=2)

def runTestKey_n(FileName, times, step, numSamples, mode):
    # keras.backend.clear_session()
    timeRec = {}
    for i in range(times):
        fname = FileName + '_{}'.format(str(i))
        exp = Explain('testFiles/' + fname, mode, model=None, step=step, numSamples=numSamples)
        st = time.time()
        resKeyword = exp.explain_n()
        et = time.time()
        tmpName = generateOutFile(fname, 'keyword_n_step{}_{}'.format(step, numSamples), resKeyword, mode)
        timeRec[tmpName] = et - st
    if mode == 111:
        with open('outputs/keywordnstep{}_{}_sbt_timer'.format(str(step), FileName), 'w') as f:
            json.dump(timeRec, f, indent=2)
    else:
        with open('outputs/keywordnstep{}_{}_timer'.format(str(step), FileName), 'w') as f:
            json.dump(timeRec, f, indent=2)

def runTestLine(FileName, times, mode):
    keras.backend.clear_session()
    timeRec = {}
    for i in range(times):
        fname = FileName + '_{}'.format(str(i))
        exp = Explain('testFiles/' + fname, mode, model=None)
        st = time.time()
        resKeyword = exp.explainLine()
        et = time.time()
        tmpName = generateOutFile(fname, 'line', resKeyword, mode)
        timeRec[tmpName] = et - st
    with open('outputs/line_{}_timer'.format(FileName), 'w') as f:
        json.dump(timeRec, f, indent=2)

def runTestLine_file(FileName, i, mode):
    keras.backend.clear_session()
    timeRec = {}
    fname = FileName + '_{}'.format(str(i))
    exp = Explain('testFiles/' + fname, mode, model=None)
    st = time.time()
    resKeyword = exp.explainLine()
    et = time.time()
    tmpName = generateOutFile(fname, 'line', resKeyword, mode)
    timeRec[tmpName] = et - st
    with open('outputs/line_{}_single_timer'.format(FileName), 'w') as f:
        json.dump(timeRec, f, indent=2)

def runTestKey_file(FileName, i, step, numSamples, mode):
    # keras.backend.clear_session()
    timeRec = {}
    fname = FileName + '_{}'.format(str(i))
    exp = Explain('testFiles/' + fname, mode, model=None, step=step, numSamples=numSamples)
    st = time.time()
    resKeyword = exp.explain()
    et = time.time()
    tmpName = generateOutFile(fname, 'keyword_ncodekey_step{}_{}'.format(step, numSamples), resKeyword, mode)
    timeRec[tmpName] = et - st
    if mode == 111:
        with open('outputs/keywordnstep{}_ncodekey_{}_sbt_single_timer'.format(str(step), FileName), 'w') as f:
            json.dump(timeRec, f, indent=2)
    else:
        with open('outputs/keywordnstep{}_ncodekey_{}_single_timer'.format(str(step), FileName), 'w') as f:
            json.dump(timeRec, f, indent=2)


if __name__ == '__main__':
    runTestKey_file('smallCodesTestFile_DeepCom', 2, 0.3, 100, 10001)