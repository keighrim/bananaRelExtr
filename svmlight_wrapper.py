# /usr/bin/python
# -*- coding: utf-8 -*-

"""
This program is to:
Wrap svmlight-TK module into python interface
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

# paths and external information
SVM_DATA_PATH = os.path.join(DATA_PATH, "svm")
# this file contains all relation names
RELATIONS = json.load(
    open(os.path.join(DATA_PATH, "aceRelationSet.json")))['relations']
SVM_PATH = os.path.join(PROJECT_PATH, "lib", "svmlight-TK")
SVM_TRAINER = os.path.join(SVM_PATH, "svm_learn")
SVM_CLASSIFIER = os.path.join(SVM_PATH, "svm_classify")
SVM_RESULT = os.path.join(PROJECT_PATH, "svm_predictions")


def train(label_name, train_filename):
    """
    given a label and train set, train a binary SVM classifier, save as a model file
    """
    model_filename = os.path.join(SVM_DATA_PATH, "svm_model_" + label_name)
    subprocess.check_call([SVM_TRAINER, '-t', '5', train_filename, model_filename])


def train_all():
    """go through all relation names as labels, train SVM classifiers"""
    for rel in RELATIONS:
        train_filename = os.path.join(SVM_DATA_PATH, "svm_train_" + rel)
        train(rel, train_filename)


def make_svm_feed_files(prefix, data_filename):
    """generate training file to train SVM, from ACE data using tree reader"""
    # delete file contents first
    for label in RELATIONS:
        f = open(os.path.join(SVM_DATA_PATH, "svm_{}_{}".format(prefix, label)), "w")
        f.close()

    # then open them again for ad-hoc writing
    labelled_files = {label: open(os.path.join
                                  (SVM_DATA_PATH, "svm_{}_{}".format(prefix, label)),
                                  "a")
                      for label in RELATIONS}
    with open(os.path.join(DATA_PATH, data_filename)) as data:
        cur_filename = None
        r = None

        # read up dataset
        for num, instance in enumerate(data):
            label, filename, i_line, i_start, _, _, _, _, j_line, _, j_end, _, _, _ \
                = instance.strip().split("\t")
            i_line = int(i_line)
            i_start = int(i_start)
            j_end = int(j_end)

            # load corresponding tree files
            if cur_filename != filename:
                r = document_reader.RelExtrReader(filename)
                cur_filename = filename

            # get parameters for svm_learn
            subtree = r.get_spanning_tree(i_line, i_start, j_end)
            subtree = re.sub(r"[\n\s]+", " ", str(subtree))
            words = r.get_words(i_line, i_start, j_end)
            pos = r.get_pos(i_line, i_start, j_end)

            # write put into a file
            for l, f in labelled_files.iteritems():
                # this file has a serious problem but can't specify...
                if filename.startswith("NYT20001115.2157.0439"):
                    f.write("0 |BT| |ET|\n")
                else:
                    if label == l:
                        f.write("1" + to_tree_string(subtree, words, pos))
                    else:
                        f.write("-1" + to_tree_string(subtree, words, pos))

    for label in labelled_files.keys():
        labelled_files[label].close()


def to_tree_string(subtree, words, pos):
    """tree string in svm_learn format"""
    return " |BT| {} |BT| (BOW ({} *)) |BT| (BOP ({} *)) |ET|\n".format(
        subtree, " *)(".join(words), " *)(".join(pos))


def classify(label, data_filename):
    """given a label name, load the corresponding model file and ru svm_classify"""
    model_filename = os.path.join(SVM_DATA_PATH, "svm_model_" + label)
    if not os.path.isfile(model_filename):
        return None
    subprocess.check_call([SVM_CLASSIFIER, '-v', '0', data_filename, model_filename])
    classified = []
    with open(os.path.join(SVM_PATH, SVM_RESULT)) as svm_result:
        for line in svm_result:

            # for binary representation
            # if float(line) > 0:
            #     classified.append(1)
            # else:
            #     classified.append(0)

            # to return confidence scores directly
            classified.append(float(line))

    return classified


if __name__ == '__main__':
    make_svm_feed_files("train", "rel-trainset.gold")
    make_svm_feed_files("dev", "rel-devset.gold")
    make_svm_feed_files("test", "rel-testset.gold")
    train_all()

