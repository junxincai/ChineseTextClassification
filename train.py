# -*- coding: utf-8 -*-
# caixinjun

import argparse
from sklearn import metrics
import datetime
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn import svm
import os
import warnings
warnings.filterwarnings('ignore')


def get_data(train_file):
    target = []
    data = []
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().split("\t")
            if len(line) == 1:
                continue
            target.append(int(line[0]))
            data.append(line[1])
    data = list(map(jieba.lcut, data))
    data = [" ".join(d) for d in data]
    return data, target


def train(cls, data, target, model_path):
    cls = cls.fit(data, target)
    with open(model_path, 'wb') as f:
        pickle.dump(cls, f)

def trans(data, matrix_path, stopword_path):
    with open(stopword_path, 'r', encoding='utf-8') as fs:
        stop_words = [line.strip() for line in fs.readline()]
    tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words=stop_words)
    features = tfidf.fit_transform(data)
    with open(matrix_path, 'wb') as f:
        pickle.dump(tfidf, f)
    return features


def load_models(matrix_path, model_path):
    tfidf, cls = None, None
    if os.path.isfile(model_path):
        with open(model_path, 'rb') as f:
            cls = pickle.load(f)
    if os.path.isfile(matrix_path):
        with open(matrix_path, 'rb') as f:
            tfidf = pickle.load(f)
    return tfidf, cls

def test(matrix_path, model_path, data_path, outdir):

    curr_time = datetime.datetime.now()
    time_str = curr_time.strftime("%Y-%m-%d %H-%M-%S")
    out_path = outdir + '/%s/' % time_str
    out_file = os.path.join(out_path, "results.txt")
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    data, target = get_data(data_path)
    tfidf, cls = load_models(matrix_path, model_path)
    if tfidf==None or cls==None:
        print("cannot load models........")
        return

    feature = tfidf.transform(data)
    predicted = cls.predict(feature)

    acc = metrics.accuracy_score(target, predicted)
    pre = metrics.precision_score(target, predicted)
    recall = metrics.recall_score(target, predicted)
    f1 = metrics.f1_score(target, predicted)
    fpr, tpr, thresholds = metrics.roc_curve(target, predicted)
    auc = metrics.auc(fpr, tpr)

    print("accuracy_score: ", acc)
    print("precision_score: ", pre)
    print("recall_score: ", recall)
    print("f1_score: ", f1)
    print("auc: ", auc)

    with open(out_file, 'w', encoding='utf-8') as f:
        for label in predicted:
            f.write(str(label) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default='./data/train.txt', help='training data')
    parser.add_argument('--test', type=str, default='./data/test.txt', help='test data')
    parser.add_argument('--stopwords', type=str, default='./data/hit_stopwords.txt', help='stop words')
    parser.add_argument('--model', type=str, default='./model/svm_model.pkl', help='classification model')
    parser.add_argument('--matrix', type=str, default='./model/tfidf.pkl', help='tfidf model')
    parser.add_argument('--outpath', type=str, default='./results/', help='out path')
    args = parser.parse_args()

    print("data processing.......")
    data, target = get_data(args.train)

    print("transform data.......")
    features = trans(data, args.matrix, args.stopwords)

    print("training model.......")
    cls = svm.LinearSVC()
    train(cls, features, target, args.model)

    print("test.......")
    test(args.matrix, args.model, args.test, args.outpath)

