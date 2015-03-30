#!/usr/bin/python
#compute the accuracy of an NE tagger

#usage: evaluate-head.py [gold_file][output_file]

import sys, re
import numpy as np
from svmlight_wrapper import RELATIONS

RELATIONS.append('no_rel')
rel_dict = dict([(rel, num) for num, rel in enumerate(RELATIONS)])

if len(sys.argv) != 3:
    sys.exit("usage: evaluate-head.py [gold_file][output_file]")

#gold standard file
goldfh = open(sys.argv[1], 'r')
#system output
testfh = open(sys.argv[2], 'r')

gold_tag_list = []
#gold_word_list = []
test_tag_list = []

emptyline_pattern = re.compile(r'^\s*$')

for gline in goldfh.readlines():
    if not emptyline_pattern.match(gline):
        parts = gline.split()
        #print parts
        gold_tag_list.append(parts[0])


for tline in testfh.readlines():
    if not emptyline_pattern.match(tline):
        parts = tline.split()
        #print parts
        test_tag_list.append(parts[0])

test_total = 0
gold_total = 0
correct = 0

#print gold_tag_list
#print test_tag_list

conf_mat = np.zeros([len(RELATIONS), len(RELATIONS)], dtype=int)

for i in range(len(gold_tag_list)):
    if gold_tag_list[i] != 'no_rel':
        gold_total += 1
    if test_tag_list[i] != 'no_rel':
        test_total += 1
    if gold_tag_list[i] != 'no_rel' and gold_tag_list[i] == test_tag_list[i]:
        correct += 1

    conf_mat[rel_dict[gold_tag_list[i]], rel_dict[test_tag_list[i]]] += 1



precision = float(correct) / test_total
recall = float(correct) / gold_total
f = precision * recall * 2 / (precision + recall)

with open("confusion_matrix.tsv", "w") as matrix:
    matrix.write(
        "\t{}\tsum\n".format("\t".join(RELATIONS))      # header
        + "\n".join(["{}\t{}\t{}".format(RELATIONS[num], "\t".join(map(str, list(row))), np.sum(row)) for num, row in enumerate(conf_mat)])
        + "\nsum\t{}".format("\t".join(map(str, list([np.sum(col) for col in conf_mat.T])))))       # footer

#print correct, gold_total, test_total
print 'precision =', precision, 'recall =', recall, 'f1 =', f


