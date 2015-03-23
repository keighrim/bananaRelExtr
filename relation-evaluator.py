#!/usr/bin/python
#compute the accuracy of an NE tagger

#usage: evaluate-head.py [gold_file][output_file]

import sys, re

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

for i in range(len(gold_tag_list)):
    if gold_tag_list[i] != 'no_rel':
        gold_total += 1
    if test_tag_list[i] != 'no_rel':
        test_total += 1
    if gold_tag_list[i] != 'no_rel' and gold_tag_list[i] == test_tag_list[i]:
        correct += 1


precision = float(correct) / test_total
recall = float(correct) / gold_total
f = precision * recall * 2 / (precision + recall)

#print correct, gold_total, test_total
print 'precision =', precision, 'recall =', recall, 'f1 =', f
            
    
