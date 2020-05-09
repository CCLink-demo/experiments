# calculates the pearson correlation

import json
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats.stats import pearsonr
from myutils import seq2sent, seq2word, seqList2Sent
from Exp import Explain
import pickle
import keras

# change mode file here
modelFile = ''
sbtModelFile = ''
comlen = 13
comstart = np.zeros(comlen)

# change token file here
datstok = pickle.load(open('', 'rb'), encoding='UTF-8')
comstok = pickle.load(open('', 'rb'), encoding='UTF-8')
st = comstok.w2i['<s>']

def translateBatch(codeList, mode, model, smls=None):
    inp = np.zeros((len(codeList), 100))
    for i, code in enumerate(codeList):
        for j, w in enumerate(code.split(' ')):
            if j == 100:
                break
            if w not in datstok.w2i.keys():
                inp[i][j] = 0
            else:
                inp[i][j] = datstok.w2i[w]
    coms = np.zeros((len(codeList), comlen))
    coms[:, 0] = st

    for i in range(1, comlen):
        if not mode == 'sbt':
            results = model.predict([inp, coms], batch_size=len(codeList))
        else:
            sml = np.zeros((len(codeList), 100))
            sml[:, :] = np.array(smls)
            results = model.predict([inp, coms, sml], batch_size=len(codeList))
        for c, s in enumerate(results):
            coms[c][i] = np.argmax(s)
    strComs = [i.split(' ') for i in seqList2Sent(coms, comstok)]
    transferedComs = []
    for com in strComs:
        s = ''
        for j in range(1, len(com)):
            if com[j] == '</s>':
                break
            s += com[j] + ' '
        transferedComs.append(s)
    return [i.split(' ')[: -1] for i in transferedComs]

class CalPearson():
    numDim = 300
    embFile = 'glove.6B.{}d.txt'.format(str(numDim))
    def __init__(self, result, fromFile=True, mode='keyword', precision=False):
        super().__init__()
        self.rawData = result
        if fromFile:
            self.rawData = self._loadDataFromFile(result)
        self.mode = mode
        self.emb = self._loadEmbedding(CalPearson.embFile)
        self.rawData = self._preprocess()
        if precision:
            self.model = keras.models.load_model(modelFile)
    
    def _preprocess(self):
        processed = []
        for instance in self.rawData:
            tmpData = {}
            tmpData['code'] = instance['code']
            if self.mode == 'keyword':
                tmpData['codeKeywords'] = instance['codeKeywords']
                tmpData['codeKeyIndex'] = instance['codeKeyIndex']
            tmpData['explanations'] = instance['explanations']
            processed.append(tmpData)
        return processed
    
    def _extractKeywordScores(self, exp, instance):
        maxExp = self._findMax(exp['supports'])
        complexScore = maxExp[0]
        simpleScore = self._getSimpleScore(instance, maxExp[1], exp['commentKeyword'])
        return simpleScore, complexScore

    def _extractLineScore(self, exp, instance):
        maxExp = self._findMaxLines(exp['lineWeight'])
        complexScore = maxExp[0]
        simpleScore = self._getSimpleScore(instance, maxExp[1], exp['commentKeyword'])
        return simpleScore, complexScore

    def _extractAnchorScore(self, exp, instance):
        complexScore = exp['precision']
        simpleScore = self._getSimpleScore(instance, exp['anchors'], exp['commentKeyword'])
        return simpleScore, complexScore
    
    def _extractLimeScore(self, exp, instance):
        complexScore = exp['lime'][0][1]
        simpleScore = self._getSimpleScore(instance, exp['lime'], exp['commentKeyword'])
        return simpleScore, complexScore

    @staticmethod
    def _findMaxLines(lineWeights):
        start = np.argmax(lineWeights)
        s = np.sum(lineWeights)
        retList = []
        i = start
        tmp = 0
        while i < len(lineWeights) and lineWeights[i] == lineWeights[start]:
            retList.append(i)
            tmp += lineWeights[i]
            i += 1
        score = 0
        if s:
            score = tmp / s
        return (score, retList)

    def _extractScores(self):
        # extract data from supports for pearson correlation calculation
        simpleScore = []
        complexScore = []
        for instance in self.rawData:
            for exp in instance['explanations']:
                if self.mode == 'keyword':
                    sScore, cScore = self._extractKeywordScores(exp, instance)
                elif self.mode == 'line':
                    sScore, cScore = self._extractLineScore(exp, instance)
                elif self.mode == 'anchor':
                    sScore, cScore = self._extractAnchorScore(exp, instance)
                elif self.mode == 'lime':
                    sScore, cScore = self._extractLimeScore(exp, instance)
                simpleScore.append(sScore)
                complexScore.append(cScore)
        return simpleScore, complexScore

    def loadEmbedding(self, path):
        self.emb = self._loadEmbedding(path)

    def loadData(self, result, fromFile=True):
        self.rawData = result
        if fromFile:
            self.rawData = self._loadDataFromFile(result)

    def _getSimpleScore(self, instance, codeKeys, comKey):
        if self.mode == 'keyword':
            codeVector = self._getCodeVectorFromKeyword(instance, codeKeys)
        elif self.mode == 'line':
            codeVector = self._getCodeVectorFromLines(instance, codeKeys)
        elif self.mode == 'anchor':
            codeVector = self._getVector(codeKeys)
        elif self.mode == 'lime':
            codeVector = self._getVector(codeKeys[0][0])
        comVector = self._getComVector(comKey)
        dist = pdist(np.vstack([codeVector, comVector]), 'cosine')
        # turn nan into zero
        if np.isnan(dist[0]):
            dist[0] = 0
        return float(dist[0])

    def _getComVector(self, comKey):
        wordList = comKey.split(' ')
        return self._getVector(wordList)

    def _getCodeVectorFromLines(self, instance, lineNums):
        code = instance['code']
        wordList = []
        for l in lineNums:
            words = code[l].split(' ')
            wordList += words
        return self._getVector(wordList)

    def _getCodeVectorFromKeyword(self, instance, codeKeys):
        code = instance['code']
        codeKeyIndex = instance['codeKeyIndex']
        keywordList = []
        for codeKey in codeKeys:
            keyIndex = codeKeyIndex[codeKey]
            words = code[keyIndex[0]].split(' ')
            tmpList = [words[i] for i in keyIndex[1]]
            keywordList += tmpList
        return self._getVector(keywordList)
    
    def _getVector(self, keywordList):
        vec = np.zeros(CalPearson.numDim)
        num = 0
        for word in keywordList:
            w = word.lower()
            if w in self.emb.keys():
                vec += np.array(self.emb[w])
                num += 1
        if num:
            vec = vec / num
        return vec

    @staticmethod
    def _loadDataFromFile(resultFile):
        retData = []
        with open(resultFile) as f:
            retData = json.load(f)
        return retData
    
    @staticmethod
    def _loadEmbedding(path):
        emb = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                tmpList = line.split(' ')
                word = tmpList[0]
                vec = [float(i) for i in tmpList[1: ]]
                emb[word] = vec
        return emb
    
    @staticmethod
    def _findMax(supports):
        maxReg = (0, [])
        for support in supports:
            if support[1] >= maxReg[0]:
                maxReg = (support[1], support[0])
        return maxReg

    def calPearson(self):
        x, y = self._extractScores()
        r, p = pearsonr(x, y)
        return r, p

    @staticmethod
    def generateCodeWithKeyword(_code, key, ids):
        code = [i.split(' ') for i in _code]
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
    
    @staticmethod
    def generateRevCode(_code, key, ids):
        code = [i.split(' ') for i in _code]
        for id in ids:
            codeKey = key[id]
            lineNum = codeKey[0]
            words = codeKey[1]
            for i in range(len(code[lineNum])):
                if i not in words:
                    code[lineNum][i] = '<unk>'
        retCode = ''
        for l in code:
            retCode += ' '.join(l) + ' '
        return retCode

    def generateRevCodes(self):
        retData = []
        comKeyList = []
        for instance in self.rawData:
            code = instance['code']
            codeKeyIndex = instance['codeKeyIndex']
            for exp in instance['explanations']:
                comKey = exp['commentKeyword']
                _, maxKeys = self._findMax(exp['supports'])
                retData.append(self.generateRevCode(code, codeKeyIndex, maxKeys))
                comKeyList.append(comKey)
        return retData, comKeyList
    
    def generateCodes(self):
        retData = []
        comKeyList = []
        for instance in self.rawData:
            code = instance['code']
            for exp in instance['explanations']:
                comKey = exp['commentKeyword']
                if self.mode == 'keyword':
                    generatedCode = self.generateCodeKeyword(exp, code, instance)
                elif self.mode == 'anchor':
                    generatedCode = self.generateCodeAnchor(exp, code)
                elif self.mode == 'line':
                    generatedCode = self.generateCodeLine(exp, code)
                elif self.mode == 'lime':
                    generatedCode = self.generateCodeLime(exp, code)
                retData.append(generatedCode)
                comKeyList.append(comKey)
        return retData, comKeyList

    def generateCodeLime(self, exp, code):
        lime = exp['lime']
        codeWordList = [i.split(' ') for i in code]
        keyList = [lime[i][0] for i in range(min(3, len(lime)))]
        for l in range(len(codeWordList)):
            for i, word in enumerate(codeWordList):
                if word in keyList:
                    keyList.remove(word)
                    codeWordList[l][i] = '<unk>'
        retCode = ''
        for l in code:
            retCode += ' '.join(l) + ' '
        return retCode

    def generateCodeKeyword(self, exp, code, instance):
        codeKeyIndex = instance['codeKeyIndex']
        _, maxKeys = self._findMax(exp['supports'])
        return self.generateCodeWithKeyword(code, codeKeyIndex, maxKeys)
    
    def generateCodeAnchor(self, exp, code):
        codeWordList = [i.split(' ') for i in code]
        for l in range(len(codeWordList)):
            for i, word in enumerate(codeWordList):
                if word in exp['anchors']:
                    codeWordList[l][i] = '<unk>'
        retCode = ''
        for l in code:
            retCode += ' '.join(l) + ' '
        return retCode
    
    def generateCodeLine(self, exp, code):
        _, maxLines = self._findMaxLines(exp['lineWeight'])
        codeWordList = [i.split(' ') for i in code]
        for l in maxLines:
            tmpLine = ['<unk>'] * len(codeWordList[l])
            codeWordList[l] = tmpLine
        retCode = ''
        for l in code:
            retCode += ' '.join(l) + ' '
        return retCode

    def calPrecision(self):
        codes, comKeys = self.generateCodes()
        coms = translateBatch(codes, '', self.model, smls=None)
        assert len(coms) == len(comKeys)
        s = 0
        for i, com in enumerate(coms):
            if comKeys[i] not in com:
                s += 1
        return s / len(comKeys)
    
    def calRevPrecision(self):
        codes, comKeys = self.generateRevCodes()
        coms = translateBatch(codes, '', self.model, smls=None)
        assert len(coms) == len(comKeys)
        s = 0
        for i, com in enumerate(coms):
            if comKeys in com:
                s += 1
        return s / len(comKeys)

def runTest(fileName, methodName, numbers):
    for i in range(numbers):
        fname = fileName + '_{}'.format(str(i)) + '_out_{}'.format(methodName)
        mode = methodName.split('_')[0]
        cal = CalPearson('outputs/' + fname, fromFile=True, mode=mode)
        r, p = cal.calPearson()
        print(fname, r, p)

def runPrecision(fileName, methodName, numbers):
    printList = []
    for i in range(numbers):
        keras.backend.clear_session()
        fname = fileName + '_{}'.format(str(i)) + '_out_{}'.format(methodName)
        mode = methodName.split('_')[0]
        cal = CalPearson('outputs/' + fname, fromFile=True, mode=mode, precision=True)
        p = cal.calPrecision()
        printList.append((fname, p))
    print(printList)
    return printList

def runPrecisionFile(fileName, methodName, i):
    printList = []
    keras.backend.clear_session()
    fname = fileName + '_{}'.format(str(i)) + '_out_{}'.format(methodName)
    mode = methodName.split('_')[0]
    cal = CalPearson('outputs/' + fname, fromFile=True, mode=mode, precision=True)
    p = cal.calPrecision()
    printList.append((fname, p))
    print(printList)
    return printList

def runTest_file(fileName, methodName, i):
    fname = fileName + '_{}'.format(str(i)) + '_out_{}'.format(methodName)
    mode = methodName.split('_')[0]
    cal = CalPearson('outputs/' + fname, fromFile=True, mode=mode)
    r, p = cal.calPearson()
    print(fname, r, p)