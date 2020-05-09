# implementing the baseline by modifying anchor

import anchor_base
import anchor_explanation
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
from Exp import Explain
import time

if (sys.version_info > (3, 0)):
    def unicode(s, errors=None):
        return s


def perturbSentence(text, present, n, neighbours, probaChange=0.5,
                    topN=50, forbidden=[], forbiddenTags=['PRP$'],
                    forbiddenWords=['be'],
                    pos=['NOUN', 'VERB', 'ADJ', 'ADV', 'ADP', 'DET'], useProba=True,
                    temperature=.4):
    tokens = unicode(text).split(' ')
    forbidden = set(forbidden)
    forbiddenTags = set(forbiddenTags)
    forbiddenWords = set(forbiddenWords)
    pos = set(pos)
    raw = np.zeros((n, len(tokens)), '|S80')
    data = np.ones((n, len(tokens)))
    raw[:] = [x for x in tokens]
    for i, t in enumerate(tokens):
        if i in present:
            continue

        # add more specific rules in this line
        if t not in forbiddenWords:
            r_neighbours = [
                               (unicode(x[0].text.encode('utf-8'), errors='ignore'), x[1])
                               for x in neighbours.neighbours(t)
                           ][:topN]
            if not r_neighbours:
                continue
            t_neighbours = [x[0] for x in r_neighbours]
            weights = np.array([x[1] for x in r_neighbours])
            if useProba:
                weights = weights ** (1. / temperature)
                weights = weights / sum(weights)
                raw[:, i] = np.random.choice(t_neighbours, n, p=weights, replace=True)
                data[:, i] = raw[:, i] == t.encode()
            else:
                n_changed = np.random.binomial(n, probaChange)
                changed = np.random.choice(n, n_changed, replace=False)
                if t in t_neighbours:
                    idx = t_neighbours.index(t)
                    weights[idx] = 0
                weights = weights / sum(weights)
                raw[changed, i] = np.random.choice(t_neighbours, n_changed, p=weights)
                data[changed, i] = 0
    if (sys.version_info > (3, 0)):
        raw = [' '.join([y.decode() for y in x]) for x in raw]
    else:
        raw = [' '.join(x) for x in raw]
    return raw, data

class Word:
    def __init__(self, _emb, _txt):
        self.emb = _emb
        self.text = _txt

    def similarity(self, _word):
        vectorA = np.mat(self.emb)
        vectorB = np.mat(_word.emb)
        num = float(vectorA * vectorB.T)
        denom = np.linalg.norm(vectorA) * np.linalg.norm(vectorB)
        cos = num / denom
        sim = 0.5 + 0.5 * cos
        return sim


class EMB(object):
    def __init__(self, path):
        self.path = path
        self.emb, self.vocab = self.getEmbVocab(path)

    @staticmethod
    def getEmbVocab(path):
        file = open(path, 'r', encoding='utf-8')
        emb = json.load(file)
        vocab = {word: Word(emb[word], word) for word in emb.keys()}
        return emb, vocab

    def tokenize(self, _string):
        words = _string.split(' ')
        retList = []
        for w in words:
            if w in self.vocab:
                retList.append(self.emb[w])
            else:
                retList.append(self.emb[0])


class MyNeighbours:
    def __init__(self, _emb):
        self.emb = _emb
        self.n = {}

    def neighbours(self, word):
        word = unicode(word)
        orig_word = word
        if word not in self.n:
            if word not in self.emb.vocab:
                self.n[word] = []
            else:
                word = self.emb.vocab[unicode(word)]
                queries = [w[1] for w in self.emb.vocab.items()]
                queries += [word]
                by_similarity = sorted(
                    queries, key=lambda w: word.similarity(w), reverse=True
                )
                self.n[orig_word] = [(Word(self.emb.emb[w.text], w.text), word.similarity(w))
                                     for w in by_similarity]

        return self.n[orig_word]


class AnchorSeq2Seq(object):
    def __init__(self, _emb, _classNames, _useUNKDistribution=True, _maskString='UNK'):
        self.emb = _emb
        self.classNames = _classNames
        self.neighbours = MyNeighbours(self.emb)
        self.useUNKDistribution = _useUNKDistribution
        self.maskString = _maskString

    def getSampleFn(self, text, classifierFn, useProba=False):
        trueLabel = classifierFn([text])[0]
        processed = text.split(' ')
        words = [w for w in processed]
        positions = [i for i, _ in enumerate(processed)]

        def sampleFn(present, numSamples, compute_labels=True):
            # print(numSamples)
            if self.useUNKDistribution:
                data = np.ones((numSamples, len(words)))
                raw = np.zeros((numSamples, len(words)), '|S80')
                raw[:] = words
                for i, t in enumerate(words):
                    if i in present:
                        continue
                    nChanged = np.random.binomial(numSamples, .5)
                    changed = np.random.choice(numSamples, nChanged, replace=False)
                    raw[changed, i] = self.maskString
                    data[changed, i] = 0
                if (sys.version_info > (3, 0)):
                    rawData = [' '.join([y.decode() for y in x]) for x in raw]
                else:
                    rawData = [' '.join(x) for x in raw]
            else:
                rawData, data = perturbSentence(
                    text, present, numSamples, self.neighbours, topN=100,
                    useProba=useProba
                )
            labels = []
            if compute_labels:
                labels = (classifierFn(rawData) == trueLabel).astype(int)
            labels = np.array(labels)
            rawData = np.array(rawData).reshape(-1, 1)
            return rawData, data, labels

        return words, positions, trueLabel, sampleFn

    def explainInstance(self, text, classifierFn, threshold=0.95,
                        delta=0.1, tau=0.15, batchSize=100, useProba=False,
                        beamSize=4, **kwargs):
        words, positions, trueLabel, sampleFn = self.getSampleFn(text, classifierFn, useProba=useProba)
        exp = anchor_base.AnchorBaseBeam.anchor_beam(
            sampleFn, delta=delta, epsilon=tau, batch_size=batchSize,
            desired_confidence=threshold, stop_on_first=True, **kwargs
        )
        exp['names'] = [words[x] for x in exp['feature']]
        exp['positions'] = [positions[x] for x in exp['feature']]
        exp['instance'] = text
        exp['prediction'] = trueLabel
        explanation = anchor_explanation.AnchorExplanation('text', exp, self.as_html)
        return explanation

    def as_html(self, exp):
        predict_proba = np.zeros(len(self.class_names))
        exp['prediction'] = int(exp['prediction'])
        predict_proba[exp['prediction']] = 1
        predict_proba = list(predict_proba)

        def jsonize(x):
            return json.dumps(x)

        this_dir, _ = os.path.split(__file__)
        bundle = open(os.path.join(this_dir, 'bundle.js'), encoding='utf8').read()
        random_id = 'top_div' + id_generator()

        example_obj = []

        def process_examples(examples, idx):
            idxs = exp['feature'][:idx + 1]
            out_dict = {}
            new_names = {'covered_true': 'coveredTrue', 'covered_false': 'coveredFalse', 'covered': 'covered'}
            for name, new in new_names.items():
                ex = [x[0] for x in examples[name]]
                out = []
                for e in ex:
                    processed = self.nlp(unicode(str(e)))
                    raw_indexes = [(processed[i].text, processed[i].idx, exp['prediction']) for i in idxs]
                    out.append({'text': e, 'rawIndexes': raw_indexes})
                out_dict[new] = out
            return out_dict

        example_obj = []
        for i, examples in enumerate(exp['examples']):
            example_obj.append(process_examples(examples, i))

        explanation = {'names': exp['names'],
                       'certainties': exp['precision'] if len(exp['precision']) else [exp['all_precision']],
                       'supports': exp['coverage'],
                       'allPrecision': exp['all_precision'],
                       'examples': example_obj}
        processed = self.nlp(unicode(exp['instance']))
        raw_indexes = [(processed[i].text, processed[i].idx, exp['prediction'])
                       for i in exp['feature']]
        raw_data = {'text': exp['instance'], 'rawIndexes': raw_indexes}
        jsonize(raw_indexes)

        out = u'''<html>
        <meta http-equiv="content-type" content="text/html; charset=UTF8">
        <head><script>%s </script></head><body>''' % bundle
        out += u'''
        <div id="{random_id}" />
        <script>
            div = d3.select("#{random_id}");
            lime.RenderExplanationFrame(div,{label_names}, {predict_proba},
            {true_class}, {explanation}, {raw_data}, "text", "anchor");
        </script>'''.format(random_id=random_id,
                            label_names=jsonize(self.class_names),
                            predict_proba=jsonize(list(predict_proba)),
                            true_class=jsonize(False),
                            explanation=jsonize(explanation),
                            raw_data=jsonize(raw_data))
        out += u'</body></html>'
        return out

    def show_in_notebook(self, exp, true_class=False, predict_proba_fn=None):
        """Bla"""
        out = self.as_html(exp, true_class, predict_proba_fn)
        from IPython.core.display import display, HTML
        display(HTML(out))

# change the model file here
modelfile = ''

# change the dataset token here
datstok = pickle.load(open('', 'rb'), encoding='UTF-8')
comstok = pickle.load(open('', 'rb'), encoding='UTF-8')
smltok = pickle.load(open('', 'rb'), encoding='utf-8')
model = keras.models.load_model(modelfile)

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

def predictor(codeList):
    retList = []
    inpts = np.zeros((len(codeList), 100))
    coms = np.zeros((len(codeList), comlen))
    coms[:,0] = st

    for i, c in enumerate(codeList):
        for j, w in enumerate(c.split(' ')):
            if w not in datstok.w2i.keys():
                inpts[i][j] = 0
            else:
                inpts[i][j] = datstok.w2i[w]
    for i in range(1, comlen):
        results = model.predict([inpts, coms], batch_size=len(codeList))
        for c, s in enumerate(results):
            coms[c][i] = np.argmax(s)
    
    for c in coms:
        if key in seq2sent(c, comstok).split(' '):
            retList.append(1)
        else:
            retList.append(0)
    return np.array(retList)

def finalExplain(codes):
    resData = []
    r = Rake()
    classNames = ['negative', 'positive']
    emb = EMB('emb.json')
    exp = AnchorSeq2Seq(emb, ['negative', 'positive'], _useUNKDistribution=True)
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
                explanation = exp.explainInstance(code, predictor)
                tmpExp['anchors'] = explanation.names()
                tmpExp['precision'] = explanation.precision()
                tmpList.append(tmpExp)
            tmpResult['explanations'] = tmpList
            resData.append(tmpResult)
            bar.update(j)
    return resData

def generateOutFile(intputFile, method, data):
    fname = intputFile + '_out_{}'.format(method)
    with open('outputs/' + fname, 'w') as f:
        json.dump(data, f, indent=2)
    return fname

def runTest(fileName, times, mode):
    timRec = {}
    for i in range(times):
        fname = fileName + '_{}'.format(str(i))
        pre = Explain('testFiles/' + fname, mode, model=model)
        codes = [s[0] for s in pre.tokenedCodes]
        st = time.time()
        result = finalExplain(codes)
        et = time.time()
        tmpName = generateOutFile(fname, 'anchor_500', result)
        timRec[tmpName] = et - st
    with open('outputs/anchor_{}_500_timer'.format(fileName), 'w') as f:
        json.dump(timRec, f, indent=2)

def runTest_file(fileName, i, mode):
    timRec = {}
    fname = fileName + '_{}'.format(str(i))
    pre = Explain('testFiles/' + fname, mode, model=model)
    codes = [s[0] for s in pre.tokenedCodes]
    st = time.time()
    result = finalExplain(codes)
    et = time.time()
    tmpName = generateOutFile(fname, 'anchor', result)
    timRec[tmpName] = et - st
    with open('outputs/anchor_{}_single_timer'.format(fileName), 'w') as f:
        json.dump(timRec, f, indent=2)


if __name__ == '__main__':
    runTest('smallCodesTestFile_DeepCom', 3, 1)
    runTest('smallCodesTestFile_OriTest', 3, 101)
    runTest('smallCodesTestFile_OriTrain', 3, 101)