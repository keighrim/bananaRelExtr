# /usr/bin/python
# -*- coding: utf-8 -*-

"""
This program is to:

"""
import re
import sys

reload(sys)
sys.setdefaultencoding('utf8')

__author__ = 'krim'
__date__ = '3/27/15'
__email__ = 'krim@brandeis.edu'

import os
import subprocess
import json
import document_reader
from document_reader import PROJECT_PATH
from document_reader import DATA_PATH

RES_PATH = os.path.join(PROJECT_PATH, "resources")
RELATIONS = json.load(
    open(os.path.join(RES_PATH, "aceRelationSet.json")))['relations']
SVM_PATH = os.path.join(PROJECT_PATH, "lib", "svmlight-TK")
SVM_TRAINER = os.path.join(SVM_PATH, "svm_learn")
SVM_CLASSIFIER = os.path.join(SVM_PATH, "svm_classify")
SVM_RESULT = os.path.join(PROJECT_PATH, "svm_predictions")


def train(label_name, train_filename):
    model_filename = os.path.join(RES_PATH, "svm_model_" + label_name)
    subprocess.check_call([SVM_TRAINER, '-t', '5', train_filename, model_filename])


def train_all():
    for rel in RELATIONS:
        train_filename = os.path.join(RES_PATH, "svm_train_" + rel)
        train(rel, train_filename)


def make_svm_feed_files(prefix, data_filename):
    # delete file contents first
    for label in RELATIONS:
        f = open(os.path.join(RES_PATH, "svm_{}_{}".format(prefix, label)), "w")
        f.close()

    # then open them again for ad-hoc writing
    labelled_files = {label: open(os.path.join
                                  (RES_PATH, "svm_{}_{}".format(prefix, label)),
                                  "a")
                      for label in RELATIONS}
    with open(os.path.join(DATA_PATH, data_filename)) as data:
        cur_filename = None
        r = None

        for num, instance in enumerate(data):

            # i_line, i_start, i_end, i_ner, _, i_word, \
            # j_line, j_start, j_end, j_ner, _, j_word, \
            label, filename, i_line, i_start, _, _, _, _, j_line, _, j_end, _, _, _ \
                = instance.strip().split("\t")
            i_line = int(i_line)
            i_start = int(i_start)
            j_end = int(j_end)

            # this file has a serious problem but can't specify...
            if filename.startswith("NYT20001115.2157.0439"):
                continue


            if cur_filename != filename:
                r = document_reader.RelExtrReader(filename)
                cur_filename = filename

            subtree = r.get_spanning_tree(i_line, i_start, j_end)
            subtree = re.sub(r"[\n\s]+", " ", str(subtree))
            subtree = subtree.replace("--", "-")
            words = r.get_words(i_line, i_start, j_end)
            pos = r.get_pos(i_line, i_start, j_end)

            for l, f in labelled_files.iteritems():
                if label == l:
                    f.write("1" + to_tree_string(subtree, words, pos))
                else:
                    f.write("-1" + to_tree_string(subtree, words, pos))

    for label in labelled_files.keys():
        labelled_files[label].close()


def to_tree_string(subtree, words, pos):
    return " |BT| {} |BT| (BOW ({} *)) |BT| (BOP ({} *)) |ET|\n".format(
        subtree, " *)(".join(words), " *)(".join(pos))


def classify(label, data_filename):
    model_filename = "svm_model_" + label
    subprocess.check_call([SVM_CLASSIFIER, '-t', '5', data_filename, model_filename])
    classified = []
    with open(os.path.join(SVM_PATH, SVM_RESULT)) as svm_result:
        for line in svm_result:
            if float(line) > 0:
                classified.append("T")
            else:
                classified.append("F")
    return classified


if __name__ == '__main__':
    # make_svm_feed_files("train", "rel-trainset.gold")
    # make_svm_feed_files("dev", "rel-devset.gold")
    # make_svm_feed_files("test", "rel-testset.gold")
    train_all()

