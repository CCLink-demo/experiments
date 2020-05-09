# map the relationship to the original code

import json
from myTokenizer import Token

token = Token()

def squeezeSpaces(code):
    retCode = []
    for line in code:
        l = line.split(' ')
        l = ' '.join(l)
        retCode.append(l)
    return retCode

def mergeToken(keyIndex, relation, tokenedCode):
    tokens = tokenedCode[keyIndex[0]].split(' ')
    rel = relation[keyIndex[0]]
    indexes = keyIndex[1]
    s = tokens[indexes[0]]
    for i in range(1, len(indexes)):
        if rel[indexes[i - 1]] == rel[indexes[i]]:
            s += tokens[indexes[i]]
        else:
            s += ' {}'.format(tokens[indexes[i]])
    return s

def findLineRel(oldLine, newLine):
    newIndex = 0
    rel = []
    for line in oldLine:
        while newIndex < len(newLine):
            if newLine[newIndex].find(line) >= 0:
                rel.append(newIndex)
                newIndex += 1
                break
            newIndex += 1
    assert len(rel) == len(oldLine)
    return rel

def fineMatchMerged_n(line, KeywordStr):
    retList = []
    keywordList = KeywordStr.split(' ')
    wordList = line.split(' ')
    lenOfWords = [len(w) for w in wordList]
    stIndexes = [0] * len(wordList)
    for i in range(1, len(wordList)):
        stIndexes[i] = stIndexes[i - 1] + lenOfWords[i - 1] + 1
    print(stIndexes)
    print(line)
    print(KeywordStr)
    for i in range(len(wordList) - 1, -1, -1):
        tmpList = []
        success = True
        t = i
        for j in range(len(keywordList) - 1, -1, -1):
            print(wordList[t])
            print(keywordList[j])
            index = wordList[t].find(keywordList[j])
            if j == len(keywordList) - 1 and index != 0 and len(keywordList) != 1:
                success = False
                break
            if index + len(keywordList[j]) != len(wordList[t]) and len(keywordList) != 1:
                success = False
                break
            print(stIndexes[t], len(keywordList[j]))
            tmpList.append([stIndexes[t], len(keywordList[j])])
            t -= 1
            while not wordList[t].isalpha() and t >= 0:
                if len(wordList[t]) == 0:
                    break
                if wordList[t][-1] == '_':
                    wordList[t] = wordList[t][:-1]
                    break
                t -= 1
            # input()
        if success:
            retList = tmpList
            break
    return retList


def fineMatchMerged(line, KeywordStr):
    retList = []
    wordList = KeywordStr.split(' ')
    tmpLine = line
    cutted = 0
    for word in wordList:
        index = tmpLine.find(word)
        tmpLine = tmpLine[index + len(word): ]
        retList.append([index + cutted, len(word)])
        cutted += index + len(word)
    return retList
        

def matchMerged(line, KeywrodStr):
    index = line.find(KeywrodStr)
    # assert index >= 0 and index < len(KeywrodStr)
    if index < 0 or index >= len(line):
        return [0, len(line)]
    return [index, len(KeywrodStr)]

def mySplitStr(string, splitter):
    tmpString = ''
    for char in string:
        tmpString += char
        if char in splitter:
            tmpString += '\n'
    if tmpString[-1] == '\n':
        tmpString = tmpString[: -1]
    return tmpString.split('\n')[: -1]

def mySplitCodes(codes):
    retCodes = []
    for code in codes:
        splitted = mySplitStr(code[0], ['{', '}', ';'])
        tmpList = []
        for line in splitted:
            if not line.isspace():
                tmpList.append(line)
        retCodes.append(tmpList)
    return retCodes

def Check(lined1, lined2):
    s = 0
    for i in range(100):
        rel = findLineRel(lined1[i], lined2[i])
        if len(lined1[i]) != len(lined2[i]):
            print(lined1[i], len(lined1[i]))
            print(lined2[i], len(lined2[i]))
            print(rel)
            s += 1
    print(s)

def getRel(instances, codes):
    linedCodes = token.generateLinesFromRawCodes(codes)
    nLinedCodes = mySplitCodes(codes)
    for i, instance in enumerate(instances):
        code = linedCodes[i]
        ncode = nLinedCodes[i]
        t, r = token.getFromLinedCode_n(code)
        code = squeezeSpaces(code)
        lineRel = findLineRel(code, ncode)
        nKeyIndex = []
        nCodeKey = []
        codeKeyIndexes = instance['codeKeyIndex']
        for keyIndex in codeKeyIndexes:
            merged = mergeToken(keyIndex, r, t)
            indexes = [lineRel[keyIndex[0]], fineMatchMerged_n(ncode[lineRel[keyIndex[0]]], merged)]
            nKeyIndex.append(indexes)
            nCodeKey.append(merged)
        instance['ncode'] = ncode
        instance['ncodeKeyword'] = nCodeKey
        instance['ncodeKeyIndex'] = nKeyIndex
    return instances

def generateNewFiles(fileName, mode):
    for i in range(3):
        fname = 'outputs/' + fileName + '_{}_out_keyword_step0.3'.format(str(i))
        if i == 2 and mode == 'small':
            fname = 'outputs/' + fileName + '_{}_out_keyword_n_step0.3_100'.format(str(i))
        codeFile = 'testFiles/' + fileName + '_{}'.format(str(i))
        cf = open(codeFile)
        codes = [[i] for i in cf.readlines()]
        cf.close()
        outFileName = fname + '_newnew.json'

        with open(fname) as f:
            instances = json.load(f)
            instances = getRel(instances, codes)
            of = open(outFileName, 'w')
            json.dump(instances, of, indent=2)

if __name__ == '__main__':
    generateNewFiles('longCodeTestFile_DeepCom', 'long')