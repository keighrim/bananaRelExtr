# /usr/bin/python
# -*- coding: utf-8 -*-

"""
This program is to:
Use various feature functions to train a MALLET MaxEnt model to classify
various relations between entities in partial ACE dataset
"""
import collections
import subprocess
import sys
import numpy as np
import time
import svmlight_wrapper


reload(sys)
sys.setdefaultencoding('utf8')

__author__ = ["Keigh Rim", "Todd Curcuru", "Yalin Liu"]
__date__ = "3/20/2015"
__email__ = ['krim@brandeis.edu', 'tcurcuru@brandeis.edu', 'yalin@brandeis.edu']

import os
import document_reader

PROJECT_PATH = os.getcwd()
DATA_PATH = os.path.join(PROJECT_PATH, "data")


class FeatureTagger():
    """FeatureTagger is a framework for tagging tokens from data file"""
    T = 'T'
    F = 'F'

    def __init__(self):
        # all feature_functions should
        # 1. take no parameters (use self.pairs)
        # 2. return a list or an iterable which has len of # number of tokens
        self.feature_functions = [

            #####################
            # Features from PA2 #
            #####################

            # string
            self.string_match_no_articles,
            self.str_stem_match,
            self.words_str_match,
            self.acronym_match,
            # self.string_contains_no_articles,         # hurts
            self.word_overlap,

            # entity level
            self.j_pronoun,
            self.only_i_pronoun,
            self.i_proper,
            self.j_proper,
            self.both_proper,
            self.i_proper_j_pronoun,
            # self.j_definite,                          # hurts
            self.j_demonstrative,
            self.pro_str_match,
            self.pn_str_match,
            self.both_diff_proper,
            self.pn_str_contains,
            self.i_pronoun,
            self.only_j_pronoun,

            # ner tag
            self.ner_tag_match,

            # position
            self.distance_sent,
            self.in_same_sent,
            self.i_precedes_j,

            # agreement
            self.number_agree,

            # syntactic tree
            self.distance_tree,
            self.distance_tree_sum,

            # dependency tree
            self.i_object,
            self.j_object,
            self.both_object,
            self.i_subject,
            self.j_subject,
            self.both_subject,
            # self.share_governing_verb,                # hurts
            self.governing_verbs_share_synset,
            self.syn_verb_same_role,
            self.appositive,
            # THREE features hurt; best perfomance using feats from PA2 is:
            # p = 0.220338983051 r = 0.0249520153551 f1 = 0.0448275862069

            ################
            # new features #
            ################

            # words
            self.i_head_words,
            self.j_head_words,
            self.i_words,
            self.j_words,

            # ner tag
            self.i_ner_tag,
            self.j_ner_tag,

            # dependency tree
            self.rels_i_to_lca,
            self.rels_j_to_lca,
            self.rels_between_i_j,

            # context features
            # self.bag_of_words_between,          #30 min train/test --> hurts
            self.pos_between,
            self.no_words_between,
            # self.i_prev_word,         # hurts
            # self.i_prev_word_2,       # hurts
            # self.i_prev_word_3,       # hurts
            self.i_prev_pos,
            # self.i_prev_pos_2,        # hurts
            # self.i_prev_pos_3,        # hurts
            # self.j_prev_word,         # hurts
            # self.j_prev_word_2,       # hurts
            # self.j_prev_word_3,       # hurts
            # self.j_prev_pos,          # hurts
            # self.j_prev_pos_2,        # hurts
            # self.j_prev_pos_3,        # hurts
            # self.i_next_word_2,       # hurts
            # self.i_next_word_3,       # hurts
            self.i_next_pos,
            # self.i_next_pos_2,        # hurts
            # self.i_next_pos_2,        # hurts
            # self.j_next_word,         # hurts
            # self.j_next_word_2,       # hurts
            # self.j_next_word_3,       # hurts
            # self.j_next_pos,          # hurts
            # self.j_next_pos_2,        # hurts
            # self.j_next_pos_2,        # hurts

            self.words_between,
            # self.first_word_between,  # hurts
            # self.last_word_between,   # hurts

            # parse tree
            self.nodes_i_to_lca,
            # self.nodes_j_to_lca,      # hurts
            self.nodes_i_to_j,
            # self.nonterminals_i_to_lca,   # hurts
            # self.nonterminals_j_to_lca,   # hurts
            # self.nonterminals_i_to_j,     # hurts
            # self.nodes_i_to_j_collapsed,  # hurts
            # self.nonterminals_i_to_j_collapsed, # hurts
            # self.lca_nodename,            # hurts

            # tree kernel
            # self.take_svm_tk_results,     # hurts
        ]

        self.load_dicts()
        self.pairs = None

    def read_input_data(self, input_filename, labeled=True):
        """load sentences from data file"""
        self.pairs = []
        self.svm_prefix = input_filename.split(os.sep)[-1].split(".")[0][4:-3]
        cur_filename = None
        with open(os.path.join(DATA_PATH, input_filename)) as in_file:
            for line in in_file:
                if labeled:
                    gold_label, filename, \
                    i_line, i_start, i_end, i_ner, _, i_word, \
                    j_line, j_start, j_end, j_ner, _, j_word, \
                        = line.strip().split("\t")
                else:
                    filename, \
                    i_line, i_start, i_end, i_ner, _, i_word, \
                    j_line, j_start, j_end, j_ner, _, j_word, \
                        = line.strip().split("\t")
                    gold_label = ""

                # split underscored words
                i_words = i_word.split("_")
                j_words = j_word.split("_")

                if cur_filename != filename:
                    r = document_reader.RelExtrReader(filename)
                    cur_filename = filename
                i_pos = r.get_pos(i_line, i_start, i_end)
                j_pos = r.get_pos(j_line, j_start, j_end)
                pair = [
                    # info on i                                         # idx 0
                    (i_words,                       # 0: list
                     i_pos,                         # 1: list
                     i_ner,                         # 2: str
                     int(i_line),                   # 3: int
                     (int(i_start), int(i_end))),   # 4: tuple(int, int)
                    # info on j                                             # 1
                    (j_words, j_pos, j_ner, int(j_line), (int(j_start), int(j_end))),
                    # additional info
                    gold_label.strip(),                                     # 2
                    i_line == j_line,                                       # 3
                    filename                                                # 4
                ]
                try:
                    assert j_words == r.get_words(j_line, j_start, j_end)
                except AssertionError:
                    print "mismatch of I at file {} line {} tokens {}-{}, " \
                          "rawtext: {}, input_data: {}".format(
                        filename, j_line, j_start, j_end,
                        r.get_words(j_line, j_start, j_end), j_words)
                try:
                    assert i_words == r.get_words(i_line, i_start, i_end)
                except AssertionError:
                    print "mismatch of I at file {} line {} tokens {}-{}, " \
                          "rawtext: {}, input_data: {}".format(
                        filename, i_line, i_start, i_end,
                        r.get_words(i_line, i_start, i_end), i_words)
                self.pairs.append(pair)
        print "INSTANCES: ", len(self.pairs)

    def load_dicts(self):
        """
        Creates a dict of all words in training data. Each word has a unique index
        """
        words, tags = document_reader.get_all_words_and_postags()
        self.w_dict = {}
        for num, w in enumerate(words):
            self.w_dict[w] = num
        self.pos_dict = {}
        for num, p in enumerate(tags):
            self.pos_dict[p] = num

    def is_coref(self):
        """return gold standard labels for each pairs"""
        return [p[2] for p in self.pairs]

    def get_i_words(self):
        """Return list of i words"""
        return [p[0][0] for p in self.pairs]

    def get_j_words(self):
        """Return list of j words"""
        return [p[1][0] for p in self.pairs]

    def get_i_poss(self):
        """Return list of pos tags of i words"""
        return [p[0][1] for p in self.pairs]

    def get_j_poss(self):
        """Return list of pos tags of j words"""
        return [p[1][1] for p in self.pairs]

    def get_i_ners(self):
        """Return list of ner tag of i words"""
        return [p[0][2] for p in self.pairs]

    def get_j_ners(self):
        """Return list of ner tag of j words"""
        return [p[1][2] for p in self.pairs]

    def get_i_j_words(self):
        """Return zipped list of words of i mention and j mention"""
        return zip(self.get_i_words(), self.get_j_words())

    def get_i_head_idx(self):
        """Return list of head indices of i mentions"""
        return self.get_head_idx(0)

    def get_j_head_idx(self):
        """Return list of head indices of j mentions"""
        return self.get_head_idx(1)

    def get_head_idx(self, i_or_j):
        """Return list of head indices of user's chosen mention (0 / 1)"""
        idxs = []
        for pair in self.pairs:
            words = pair[i_or_j][0]
            tags = pair[i_or_j][1]
            start = pair[i_or_j][4][0]
            idxs.append(self.get_head(words, tags)[2] + start)
        return idxs
    
    @staticmethod
    def get_head(words, tags):
        """
        Simple heuristic head-finding algorithm
        last NN*/CD/JJ/RB before the first IN
        Seriously?  'Baghdad/RB?'
        :return:(head_word, head_pos, head_index)
        """
        cur = ()
        for i, (word, tag) in enumerate(zip(words, tags)):
            if tag.startswith(("NN", "PR", "CD", "JJ")):
                cur = (word, tag, i)
            elif tag == "IN":
                if cur == ():
                    return word, tag, i
                else:
                    return cur
        if cur == ():
            return words[-1], tags[-1], len(words) - 1
        else:
            return cur

    def feature_matrix(self, out_filename, train=True):
        """use this method to get all feature values and printed out as a file"""
        with open(out_filename, "w") as outf:
            features = self.get_features(train)
            for tok_index in range(len(features)):
                outf.write("\t".join(features[tok_index]) + "\n")

    def get_features(self, train=True):
        """traverse function list and get all values in a dictionary"""
        features = collections.defaultdict(list)

        # add gold bio tags while training
        if train:
            self.feature_functions.insert(0, self.is_coref)
            
        # traverse functions
        # note that all function should take no parameter and return an iterable
        # which has length of the number of total tokens
        for fn in self.feature_functions:
            for num, feature in enumerate(fn()):
                features[num].append(feature)

        # remove gold tags when it's done
        if train:
            self.feature_functions.remove(self.is_coref)
        return features

    """""""""""""""""
    feature functions
    """""""""""""""""

    def i_pronoun(self):
        """Is the first entity a pronoun"""
        name = "i_pronoun="
        values = []
        poss = self.get_i_poss()
        for pos in poss:
            if len(pos) == 1 and pos[0].startswith("PRP"):
                values.append(name + self.T)
            else:
                values.append(name + self.F)
        return values

    def j_pronoun(self):
        """Is the second entity a pronoun"""
        name = "j_pronoun="
        values = []
        poss = self.get_j_poss()
        for pos in poss:
            if len(pos) == 1 and pos[0].startswith("PRP"):
                values.append(name + self.T)
            else:
                values.append(name + self.F)
        return values

    def only_j_pronoun(self):
        """Checks if only the second entity is a pronoun, and not the first"""
        name = "only_j_pronoun="
        values = []
        for bools in zip(self.i_pronoun(), self.j_pronoun()):
            if bools[0].endswith("false") and bools[1].endswith("true"):
                values.append(name + self.T)
            else:
                values.append(name + self.F)
        return values

    def only_i_pronoun(self):
        """Checks if only the first entity is a pronoun, and not the second"""
        name = "only_i_pronoun="
        values = []
        for bools in zip(self.i_pronoun(), self.j_pronoun()):
            if bools[1].endswith("false") and bools[0].endswith("true"):
                values.append(name + self.T)
            else:
                values.append(name + self.F)
        return values

    def i_ner_tag(self):
        """return string value of NER tag of each instance"""
        name = "i_ner="
        return map(lambda x: name + x, self.get_i_ners())

    def j_ner_tag(self):
        """return string value of NER tag of each instance"""
        name = "j_ner="
        return map(lambda x: name + x, self.get_j_ners())

    def ner_tag_match(self):
        """true if two mentions share same ber tag"""
        name = "ner_tag_match="
        values = []
        for tags in zip(self.get_i_ners(), self.get_j_ners()):
            if tags[0] == tags[1]:
                values.append(name + self.T)
            else:
                values.append(name + self.F)
        return values

    @staticmethod
    def remove_articles(words, tags):
        """Removes any articles from a list of words and a list of their tags"""
        return_string = ""
        for i in range(len(words)):
            if tags[i] != "DT":
                return_string += words[i]
        return return_string

    def string_match_no_articles(self):
        """Checks to see if two entities match exactly, without articles"""
        name = "string_match_no_articles="
        values = []
        i_words = self.get_i_words()
        j_words = self.get_j_words()
        i_tags = self.get_i_poss()
        j_tags = self.get_j_poss()
        for i in range(len(i_words)):
            comparator_i = self.remove_articles(i_words[i], i_tags[i])
            comparator_j = self.remove_articles(j_words[i], j_tags[i])
            if comparator_i == comparator_j:
                values.append(name + self.T)
            else:
                values.append(name + self.F)
        return values

    def string_contains_no_articles(self):
        """Checks if one entities is contained in another, without articles"""
        name = "string_contains_no_articles="
        values = []
        i_words = self.get_i_words()
        j_words = self.get_j_words()
        i_tags = self.get_i_poss()
        j_tags = self.get_j_poss()
        for i in range(len(i_words)):
            comparator_i = self.remove_articles(i_words[i], i_tags[i])
            comparator_j = self.remove_articles(j_words[i], j_tags[i])
            if comparator_i in comparator_j or \
                            comparator_j in comparator_i:
                values.append(name + self.T)
            else:
                values.append(name + self.F)
        return values

    def str_stem_match(self):
        """Stem first, and then check string match"""
        from nltk.stem.lancaster import LancasterStemmer
        name = "str_stem_match="
        values = []
        stemmer = LancasterStemmer()
        i_words = self.get_i_words()
        j_words = self.get_j_words()
        i_tags = self.get_i_poss()
        j_tags = self.get_j_poss()
        for i in range(len(i_words)):
            comparator_i = stemmer.stem(self.remove_articles(i_words[i], i_tags[i]))
            comparator_j = stemmer.stem(self.remove_articles(j_words[i], j_tags[i]))
            if comparator_i == comparator_j:
                values.append(name + self.T)
            else:
                values.append(name + self.F)
        return values

    def pro_str_match(self):
        """Check if both entities are pronouns and they both match"""
        name = "pro_str_match="
        values = []
        i_words = self.get_i_words()
        j_words = self.get_j_words()
        pro_bools = zip(self.i_pronoun(), self.j_pronoun())
        for i in range(len(i_words)):
            if pro_bools[i][0].endswith("true") \
                    and pro_bools[i][1].endswith("true") \
                    and i_words[i] == j_words[i]:
                values.append(name + self.T)
            else:
                values.append(name + self.F)
        return values

    def pn_str_match(self):
        """Check if both entities are proper nouns and they both match"""
        name = "pn_str_match="
        values = []
        i_words = self.get_i_words()
        j_words = self.get_j_words()
        i_tags = self.get_i_poss()
        j_tags = self.get_j_poss()
        for i in range(len(i_words)):
            i_nnps = [tag for tag in i_tags[i] if tag.startswith("NNP")]
            j_nnps = [tag for tag in j_tags[i] if tag.startswith("NNP")]
            if len(i_nnps) > 0 and len(j_nnps) > 0 \
                    and " ".join(i_words[i]) == " ".join(j_words[i]):
                values.append(name + self.T)
            else:
                values.append(name + self.F)
        return values

    def pn_str_contains(self):
        """Check if both entities are proper nouns and one contains the other"""
        name = "pn_str_contains="
        values = []
        i_words = self.get_i_words()
        j_words = self.get_j_words()
        i_tags = self.get_i_poss()
        j_tags = self.get_j_poss()
        for i in range(len(i_words)):
            i_nnps = [tag for tag in i_tags[i] if tag.startswith("NNP")]
            j_nnps = [tag for tag in j_tags[i] if tag.startswith("NNP")]
            i_string = " ".join(i_words[i])
            j_string = " ".join(j_words[i])
            if len(i_nnps) > 0 and len(j_nnps) > 0 \
                    and (j_string in i_string or i_string in j_string):
                values.append(name + self.T)
            else:
                values.append(name + self.F)
        return values

    def words_str_match(self):
        """Check if both entities are not pronouns and they both match"""
        name = "words_str_match="
        values = []
        i_words = self.get_i_words()
        j_words = self.get_j_words()
        pro_bools = zip(self.i_pronoun(), self.j_pronoun())
        for i in range(len(i_words)):
            if pro_bools[i][0].endswith("false") \
                    and pro_bools[i][1].endswith("false") \
                    and " ".join(i_words[i]) == " ".join(j_words[i]):
                values.append(name + self.T)
            else:
                values.append(name + self.F)
        return values

    def j_definite(self):
        """Check if second entity is a definite NP"""
        name = "j_definite="
        values = []
        for words in self.get_j_words():
            if words[0].lower() == "the":
                values.append(name + self.T)
            else:
                values.append(name + self.F)
        return values

    def j_demonstrative(self):
        """Check if second entity is a demonstrative NP"""
        name = "j_demonstrative="
        values = []
        demons = {"these", "those", "this", "that"}
        for words in self.get_j_words():
            if words[0].lower() in demons:
                values.append(name + self.T)
            else:
                values.append(name + self.F)
        return values

    def word_overlap(self):
        """Check if entities have any words in common"""
        name = "word_overlap="
        values = []
        i_words = self.get_i_words()
        j_words = self.get_j_words()
        for i in range(len(i_words)):
            i_set = set(word.lower() for word in i_words[i])
            j_set = set(word.lower() for word in j_words[i])
            if len(i_set.intersection(j_set)) > 0:
                values.append(name + self.T)
            else:
                values.append(name + self.F)
        return values

    def i_proper(self):
        """Check if the first mention is a proper noun"""
        name = "i_proper="
        values = []
        tags = self.get_i_poss()
        for i in range(len(tags)):
            nnps = [tag for tag in tags[i] if tag.startswith("NNP")]
            if len(nnps) == len(tags[i]):
                # print i, " found NNP at first loc"
                values.append(name + self.T)
            else:
                values.append(name + self.F)
        return values

    def j_proper(self):
        """Check if the first mention is a proper noun"""
        name = "j_proper="
        values = []
        tags = self.get_j_poss()
        for i in range(len(tags)):
            nnps = [tag for tag in tags[i] if tag.startswith("NNP")]
            if len(nnps) == len(tags[i]):
                values.append(name + self.T)
            else:
                values.append(name + self.F)
        return values

    def i_proper_j_pronoun(self):
        """check if i is a proper noun && j is a pronoun"""
        name = "i_pn_i_pro="
        values = []
        for bools in zip(self.i_proper(), self.j_pronoun()):
            if bools[0].endswith("true") and bools[1].endswith("true"):
                values.append(name + self.T)
            else:
                values.append(name + self.F)
        return values

    def both_proper(self):
        """Check if both entities are proper nouns"""
        name = "both_proper="
        values = []
        for bools in zip(self.i_proper(), self.j_proper()):
            if bools[0].endswith("true") and bools[1].endswith("true"):
                values.append(name + self.T)
            else:
                values.append(name + self.F)
        return values

    def both_diff_proper(self):
        """Check if both entities are proper nouns and no words match"""
        name = "both_diff_proper="
        values = []
        i_words = self.get_i_words()
        j_words = self.get_j_words()
        i_tags = self.get_i_poss()
        j_tags = self.get_j_poss()
        for i in range(len(i_words)):
            i_nnps = [tag for tag in i_tags[i] if tag.startswith("NNP")]
            j_nnps = [tag for tag in j_tags[i] if tag.startswith("NNP")]
            i_set = set(word.lower() for word in i_words[i])
            j_set = set(word.lower() for word in j_words[i])
            if len(i_nnps) > 0 and len(j_nnps) > 0 and \
                            len(i_set.intersection(j_set)) == 0:
                values.append(name + self.T)
            else:
                values.append(name + self.F)
        return values

    def acronym_match(self):
        """Check lexically if one entity is an acronym of the other"""
        name = "acronym_match="
        values = []
        all_i_words = self.get_i_words()
        all_j_words = self.get_j_words()
        for i in range(len(self.pairs)):
            i_acronym = "".join([word[0] for word in all_i_words[i]])
            j_acronym = "".join([word[0] for word in all_j_words[i]])
            if i_acronym in all_j_words[0] or j_acronym in all_i_words[0]:
                values.append(name + self.T)
            else:
                values.append(name + self.F)
        return values

    def distance_sent(self):
        """Returns the number of sentences between two mentions"""
        name = "dist_sent="
        values = []
        i_sents = [p[0][3] for p in self.pairs]
        j_sents = [p[1][3] for p in self.pairs]
        for num, i_sent in enumerate(i_sents):
            dist = i_sent - j_sents[num]
            values.append(name + str(dist))
        return values

    def in_same_sent(self):
        """true if two mentions are in a same sentence"""
        name = "in_same_sent="
        values = []
        for pair in self.pairs:
            if pair[3]:
                values.append(name + self.T)
            else:
                values.append(name + self.F)
        return values

    def i_precedes_j(self):
        """Check if both i and j are in a same document, and i precedes j"""
        name = "i_precedes="
        values = []

        i_sents = [p[0][3] for p in self.pairs]
        i_offset = [p[0][4] for p in self.pairs]
        j_sents = [p[1][3] for p in self.pairs]
        j_offset = [p[1][4] for p in self.pairs]

        for i in range(len(self.pairs)):
            if i_sents[i] < j_sents[i]:
                values.append(name + self.T)
            elif i_sents[i] > j_sents[i]:
                values.append(name + self.F)
            # if i and j are in the same sentence
            else:
                # end of i < start of j --> i precedes
                if i_offset[i][1] <= j_offset[i][0]:
                    values.append(name + self.T)
                else:
                    values.append(name + self.F)
        return values

    def number_agree(self):
        """
        Check for POS tags to see given NE is plural.
        In case of PRP, check word themselves
        """
        name = "number_agree="
        values = []

        def is_plural(word_pos):
            if word_pos[1] in plural_tags:
                return True
            elif word_pos[1].startswith("PRP") and word_pos[0].lower() in plural_pronouns:
                return True
            else:
                return False

        i_words = self.get_i_words()
        i_tags = self.get_i_poss()
        j_words = self.get_j_words()
        j_tags = self.get_j_poss()

        plural_tags = ("NNS", "NNPS")
        plural_pronouns = ("we", "they", "you", "theirs", "themsleves",
                           "ourselves", "our", "ours", "their")

        for num in range(len(i_words)):
            if is_plural(self.get_head(i_words[num], i_tags[num])[0:2])\
                    == is_plural(self.get_head(j_words[num], j_tags[num])[0:2]):
                values.append(name + self.T)
            else:
                values.append(name + self.F)

        return values

    def distance_tree(self):
        """
        Returns the length of path between two mentions in phrase structure parse.
        :return:["distance_tree=u<#UpwardMovements>d<#DownwardMovements>"]
        If they are not in a same sentence, return 10000
        """
        name = "distance_tree="
        values = []
        cur_filename = None
        r = None
        for pair in self.pairs:
            if not pair[3]:
                values.append(name + "10000")
            else:
                filename = pair[4]
                if cur_filename != filename:
                    r = document_reader.RelExtrReader(pair[4])
                    cur_filename = filename
                preceding = min(pair[0][4] + pair[1][4])
                following = max(pair[0][4] + pair[1][4]) - 1
                path = r.compute_tree_path_length(pair[0][3], preceding, following)
                values.append(name + "u" + str(path[0]) + "d" + str(path[1]))
        return values

    def distance_tree_sum(self):
        """
        Returns the length of path between two mentions in phrase structure parse.
        :return:["distance_tree=len(full_path)"]
        If they are not in a same sentence, return 10000
        """
        name = "distance_tree_sum="
        values = []
        cur_filename = None
        r = None
        for pair in self.pairs:
            if not pair[3]:
                values.append(name + "10000")
            else:
                filename = pair[4]
                if cur_filename != filename:
                    r = document_reader.RelExtrReader(pair[4])
                    cur_filename = filename
                preceding = min(pair[0][4] + pair[1][4])
                following = max(pair[0][4] + pair[1][4]) - 1
                path = r.compute_tree_path_length(pair[0][3], preceding, following)
                values.append(name + str(sum(path)))
        return values

    def nodes_i_to_lca(self):
        """return lists of phrase structural nodes from i to lowest ancestor"""
        name = "nodes_i_to_lca="
        values = []
        for path in self.phrase_nodes_between():
            if path is None:
                values.append(name + "None")
            else:
                values.append(name + str(path[0]))
        return values

    def nodes_j_to_lca(self):
        """return lists of phrase structural nodes from j to lowest ancestor"""
        name = "nodes_j_to_lca="
        values = []
        for path in self.phrase_nodes_between():
            if path is None:
                values.append(name + "None")
            else:
                values.append(name + str(path[1]))
        return values

    def nodes_i_to_j(self):
        """return lists of phrase structural nodes from i to j"""
        name = "nodes_i_to_j="
        values = []
        for path in self.phrase_nodes_between():
            if path is None:
                values.append(name + "None")
            else:
                values.append(name + str(path[0] + path[1]))
        return values

    def nodes_i_to_j_collapsed(self):
        """nodes from i to j, collapsed duplicates"""
        name = "c_nodes_i_to_j="
        values = []
        for path in self.phrase_nodes_between():
            if path is None:
                values.append(name + "None")
            else:
                fullpath = path[0] + path[1]
                collapsed_path = [fullpath[0]]
                for i in range(1, len(fullpath)):
                    if fullpath[i-1] != fullpath[i]:
                        collapsed_path.append(fullpath[i])
                values.append(name + str(collapsed_path))
        return values

    def nonterminals_i_to_lca(self):
        """return lists of nonterminal nodes from i to LCA"""
        name = "nt_i_to_lca="
        values = []
        for path in self.phrase_nodes_between():
            if path is None:
                values.append(name + "None")
            else:
                values.append(name + str((path[0] + path[1])[1:-1]))
        return values

    def nonterminals_j_to_lca(self):
        """return lists of nonterminal nodes from j to LCA"""
        name = "nt_j_to_lca="
        values = []
        for path in self.phrase_nodes_between():
            if path is None:
                values.append(name + "None")
            else:
                values.append(name + str(path[0][1:]))
        return values
        pass

    def nonterminals_i_to_j(self):
        """return lists of nonterminal nodes from i to j"""
        name = "nt_i_to_j="
        values = []
        for path in self.phrase_nodes_between():
            if path is None:
                values.append(name + "None")
            else:
                values.append(name + str(path[1][:-1]))
        return values

    def nonterminals_i_to_j_collapsed(self):
        """nonterminal nodes from i to j, collapsed duplicates"""
        name = "nt_i_to_j="
        values = []
        for path in self.phrase_nodes_between():
            if path is None:
                values.append(name + "None")
            else:
                nonterminal_path = path[1:-1]
                if len(nonterminal_path) == 0:
                    values.append(name + "None")
                else:
                    collapsed_path = [nonterminal_path[0]]
                    for i in range(1, len(nonterminal_path)):
                        if nonterminal_path[i-1] != nonterminal_path[i]:
                            collapsed_path.append(nonterminal_path[i])
                    values.append(name + str(collapsed_path))
        return values

    def lca_nodename(self):
        """returns the node name of lowest common ancestor"""
        name = "lca_name="
        values = []
        r = None
        cur_filename = None
        i_head_idxs = self.get_head_idx(0)
        j_head_idxs = self.get_head_idx(1)
        for i, pair in enumerate(self.pairs):
            if pair[3]:
                sent = pair[0][3]
                filename = pair[4]
                if cur_filename != filename:
                    r = document_reader.RelExtrReader(filename)
                    cur_filename = filename
                values.append(name + r.get_spanning_tree(sent, i_head_idxs[i],
                                                         j_head_idxs[i]).label())
        return values

    def phrase_nodes_between(self):
        """helper function for ps parsing path features"""
        paths = []
        r = None
        cur_filename = None
        i_head_idxs = self.get_head_idx(0)
        j_head_idxs = self.get_head_idx(1)
        for i, pair in enumerate(self.pairs):
            if pair[3]:
                sent = pair[0][3]
                filename = pair[4]
                if cur_filename != filename:
                    r = document_reader.RelExtrReader(filename)
                    cur_filename = filename
                paths.append(r.get_phrase_path(sent, i_head_idxs[i], j_head_idxs[i]))
            else:
                paths.append(None)
        return paths

    def rels_i_to_lca(self):
        """return a concatenated relations from i entity to lowest common ancestor"""
        name = "rels_i_to_lca="
        values = []
        r = None
        cur_filename = None
        i_head_idxs = self.get_head_idx(0)
        j_head_idxs = self.get_head_idx(1)
        for i, pair in enumerate(self.pairs):
            if pair[3]:
                sent = pair[0][3]
                filename = pair[4]
                if cur_filename != filename:
                    r = document_reader.RelExtrReader(filename)
                    cur_filename = filename
                values.append(name + str(r.get_dep_rel_path(
                    sent, i_head_idxs[i], j_head_idxs[i])[0]))
            else:
                values.append(name + "None")
        return values

    def rels_j_to_lca(self):
        """return a concatenated relations from lowest common ancestor to j entity"""
        name = "rels_j_to_lca="
        values = []
        r = None
        cur_filename = None
        i_head_idxs = self.get_head_idx(0)
        j_head_idxs = self.get_head_idx(1)
        for i, pair in enumerate(self.pairs):
            if pair[3]:
                sent = pair[0][3]
                filename = pair[4]
                if cur_filename != filename:
                    r = document_reader.RelExtrReader(filename)
                    cur_filename = filename
                values.append(name + str(r.get_dep_rel_path(
                    sent, i_head_idxs[i], j_head_idxs[i])[1]))
            else:
                values.append(name + "None")
        return values

    def rels_between_i_j(self):
        """return a concatenated relations from i entity to lowest common ancestor"""
        name = "rels_between_i_j="
        values = []
        r = None
        cur_filename = None
        i_head_idxs = self.get_head_idx(0)
        j_head_idxs = self.get_head_idx(1)
        for i, pair in enumerate(self.pairs):
            if pair[3]:
                sent = pair[0][3]
                filename = pair[4]
                if cur_filename != filename:
                    r = document_reader.RelExtrReader(filename)
                    cur_filename = filename
                relations = r.get_dep_rel_path(
                    sent, i_head_idxs[i], j_head_idxs[i])
                if None not in relations:
                    values.append(name + "[{}-{}]".format(
                        "-".join(relations[0]), "-".join(relations[1])))
                else:
                    values.append(name + "None")
            else:
                values.append(name + "None")
        return values


    def i_subject(self):
        """returns true if i mention is subject of its sentence"""
        name = "i_subject="
        values = []
        for val in self.is_subject(0):
            if val:
                values.append(name + self.T)
            else:
                values.append(name + self.F)
        return values

    def j_subject(self):
        """returns true if j mention is subject of its sentence"""
        name = "j_subject="
        values = []
        for val in self.is_subject(1):
            if val:
                values.append(name + self.T)
            else:
                values.append(name + self.F)
        return values

    def both_subject(self):
        """return true if both i and j mentions are subject of their sentences"""
        name = "both_subject="
        values = []
        for val in zip(self.is_subject(0), self.is_subject(1)):
            if val[0] and val[1]:
                values.append(name + self.T)
            else:
                values.append(name + self.F)
        return values

    def is_subject(self, i_or_j):
        """helper function for i_subject, j_subject, both_subject"""
        values = []
        r = None
        cur_filename = None
        head_idxs = self.get_head_idx(i_or_j)
        for i, pair in enumerate(self.pairs):
            sent = pair[i_or_j][3]
            filename = pair[4]
            if cur_filename != filename:
                r = document_reader.RelExtrReader(filename)
                cur_filename = filename
            values.append(r.is_subject(sent, head_idxs[i]))
        return values

    def i_object(self):
        """returns true if i mention is object of its sentence"""
        name = "i_object="
        values = []
        for val in self.is_object(0):
            if val:
                values.append(name + self.T)
            else:
                values.append(name + self.F)
        return values

    def j_object(self):
        """returns true if j mention is object of its sentence"""
        name = "j_object="
        values = []
        for val in self.is_object(1):
            if val:
                values.append(name + self.T)
            else:
                values.append(name + self.F)
        return values

    def both_object(self):
        """return true if both i and j mentions are object of their sentences"""
        name = "both_object="
        values = []
        for val in zip(self.is_object(0), self.is_object(1)):
            if val[0] and val[1]:
                values.append(name + self.T)
            else:
                values.append(name + self.F)
        return values

    def is_object(self, i_or_j):
        """helper function for i_object, j_object, both_object"""
        values = []
        r = None
        cur_filename = None
        head_idxs = self.get_head_idx(i_or_j)
        for i, pair in enumerate(self.pairs):
            sent = pair[i_or_j][3]
            filename = pair[4]
            if cur_filename != filename:
                r = document_reader.RelExtrReader(filename)
                cur_filename = filename
            values.append(r.is_object(sent, head_idxs[i]))
        return values

    def appositive(self):
        """return true if two mentions are in appositive construction"""
        name = "appositive="
        values = []
        r = None
        cur_filename = None
        i_head_idxs = self.get_i_head_idx()
        j_head_idxs = self.get_j_head_idx()
        for i, pair in enumerate(self.pairs):
            i_sent = pair[0][3]
            j_sent = pair[1][3]
            if i_sent != j_sent:
                values.append(name + self.F)
            else:
                filename = pair[4]
                if cur_filename != filename:
                    r = document_reader.RelExtrReader(filename)
                    cur_filename = filename
                if r.get_dep_relation(i_sent, i_head_idxs[i], j_head_idxs[i]) == "appos":
                    values.append(name + self.T)
                else:
                    values.append(name + self.F)
        return values

    def share_governing_verb(self):
        """return true if two mention has a shared verb as their governor"""
        name = "share_verb="
        values = []
        cur_filename = None
        i_head_idxs = self.get_i_head_idx()
        j_head_idxs = self.get_j_head_idx()
        for i, pair in enumerate(self.pairs):
            i_sent = pair[0][3]
            j_sent = pair[1][3]
            filename = pair[4]
            if cur_filename != filename:
                r = document_reader.RelExtrReader(filename)
                cur_filename = filename
            i_verb = r.get_deprel_verb(i_sent, i_head_idxs[i])[1]
            j_verb = r.get_deprel_verb(j_sent, j_head_idxs[i])[1]
            if i_verb == j_verb and i_verb is not None:
                values.append(name + self.T)
            else:
                values.append(name + self.F)
        return values

    def governing_verbs_share_synset(self):
        """
        return true if two verbs governing two mentions
        are in a same wordnet synset
        """
        name = "share_verb_synset="
        values = []

        cur_filename = None
        i_head_idxs = self.get_i_head_idx()
        j_head_idxs = self.get_j_head_idx()
        for i, pair in enumerate(self.pairs):
            i_sent = pair[0][3]
            j_sent = pair[1][3]
            filename = pair[4]
            if cur_filename != filename:
                r = document_reader.RelExtrReader(filename)
                cur_filename = filename
            i_verb = r.get_deprel_verb(i_sent, i_head_idxs[i])[1]
            j_verb = r.get_deprel_verb(j_sent, j_head_idxs[i])[1]
            if r.in_same_synsets(i_verb, j_verb):
                values.append(name + self.T)
            else:
                values.append(name + self.F)
        return values

    def syn_verb_same_role(self):
        """
        return true if two verbs governing two mentions
        are in a same wordnet synset
        and two mentions play same syntactic role (sbj, obj)
        """
        name = "share_verb_synset_role="
        values = []

        cur_filename = None
        i_head_idxs = self.get_i_head_idx()
        j_head_idxs = self.get_j_head_idx()
        for i, pair in enumerate(self.pairs):
            i_sent = pair[0][3]
            j_sent = pair[1][3]
            filename = pair[4]
            if cur_filename != filename:
                r = document_reader.RelExtrReader(filename)
                cur_filename = filename
            i_rel, i_verb = r.get_deprel_verb(i_sent, i_head_idxs[i])
            j_rel, j_verb = r.get_deprel_verb(j_sent, j_head_idxs[i])
            if r.in_same_synsets(i_verb, j_verb) and i_rel == j_rel:
                values.append(name + self.T)
            else:
                values.append(name + self.F)
        return values

    def bag_of_words_between(self):
        """A feature that lists all the words between two mentions"""
        name = "w_btw="
        values = []
        cur_filename = None
        r = None
        for pair in self.pairs: 
            if not pair[3]:
                values.append(name + self.F)
            else:
                filename = pair[4]
                if cur_filename != filename:
                    r = document_reader.RelExtrReader(pair[4])
                    cur_filename = filename
                preceding = min(pair[0][4] + pair[1][4])
                following = max(pair[0][4] + pair[1][4]) - 1
                words = set(r.get_words(pair[0][3], preceding, following))
                pair_values = np.zeros(len(self.w_dict), dtype=np.int)
                for idx in [self.w_dict[w] for w in words]:
                    pair_values[idx] = 1
                values.append("".join(map(str, map(int, pair_values))))
        return values

    def pos_between(self):
        """
        A feature that lists all the POS of words between two mentions
        Because of space, list is a binary number, each digit indexed to a POS tag
        """
        name = "pos_between="
        values = []
        cur_filename = None
        r = None
        for pair in self.pairs: 
            if not pair[3]:
                values.append(name + self.F)
            else:
                filename = pair[4]
                if cur_filename != filename:
                    r = document_reader.RelExtrReader(pair[4])
                    cur_filename = filename
                preceding = min(pair[0][4] + pair[1][4])
                following = max(pair[0][4] + pair[1][4]) - 1
                tags = r.get_pos(pair[0][3], preceding, following)
                pair_values = np.zeros(len(self.pos_dict), dtype=np.int)
                for idx in [self.pos_dict[p] for p in tags]:
                    pair_values[idx] = 1
                value = name + str(
                    sum(map(lambda (x, y): 10 ** y * x,
                            zip(pair_values, range(len(pair_values)-1, -1, -1))))).\
                    zfill(len(pair_values))
                values.append(value)
        return values

    def i_words(self):
        """returns bags of words of i entity"""
        name = "i_words="
        values = []
        for pair in self.pairs:
            values.append(name + str(pair[0][0]))
        return values

    def j_words(self):
        """returns bags of words of j entity"""
        name = "j_words="
        values = []
        for pair in self.pairs:
            values.append(name + str(pair[1][0]))
        return values

    def i_head_words(self):
        """returns word string of head of i entity"""
        return self.head_words("i_head=", 0)

    def j_head_words(self):
        """returns word string of head of j entity"""
        return self.head_words("i_head=", 1)

    def head_words(self, name, i_or_j):
        """returns word string of head of i entity"""
        values = []
        head_idxs = self.get_head_idx(i_or_j)
        for num, pair in enumerate(self.pairs):
            words = pair[i_or_j][0]
            start_idx = pair[i_or_j][4][0]
            head_idx = head_idxs[num]
            values.append(name + words[head_idx - start_idx])
        return values

    def prev_or_next(self, name, i_or_j, n, pos=False):
        """Helper function to return prev/next words or pos tags"""
        values = []
        cur_filename = None
        for pair in self.pairs: 
            filename = pair[4]
            if cur_filename != filename:
                r = document_reader.RelExtrReader(pair[4])
                cur_filename = filename
                
            orig_start, orig_end = pair[i_or_j][4]
            if n < 0:       
                index = orig_start + n
            else:
                index = orig_end + n
            try:
                if pos:
                    target = r.get_pos(pair[0][3], index, index + 1)
                else:
                    target = r.get_words(pair[0][3], index, index + 1)
                values.append(name + target[0])
            except IndexError:
                values.append(name + "out_of_bounds")
        return values
        
    def i_prev_word(self):
        return self.prev_or_next("i_prev_word=", 0, -1)
    
    def i_prev_word_2(self):
        return self.prev_or_next("i_prev_word_2=", 0, -2)
        
    def i_prev_word_3(self):
        return self.prev_or_next("i_prev_word_3=", 0, -3)
        
    def i_prev_pos(self):
        return self.prev_or_next("i_prev_pos=", 0, -1, True)
    
    def i_prev_pos_2(self):
        return self.prev_or_next("i_prev_pos_2=", 0, -2, True)
        
    def i_prev_pos_3(self):
        return self.prev_or_next("i_prev_pos_3=", 0, -3, True)
        
    def j_prev_word(self):
        return self.prev_or_next("j_prev_word=", 1, -1)
    
    def j_prev_word_2(self):
        return self.prev_or_next("j_prev_word_2=", 1, -2)
        
    def j_prev_word_3(self):
        return self.prev_or_next("j_prev_word_3=", 1, -3)
        
    def j_prev_pos(self):
        return self.prev_or_next("j_prev_pos=", 1, -1, True)
    
    def j_prev_pos_2(self):
        return self.prev_or_next("j_prev_pos_2=", 1, -2, True)
        
    def j_prev_pos_3(self):
        return self.prev_or_next("j_prev_pos_3=", 1, -3, True)
        
    def i_next_word(self):
        return self.prev_or_next("i_next_word=", 0, 1)
    
    def i_next_word_2(self):
        return self.prev_or_next("i_next_word_2=", 0, 2)
        
    def i_next_word_3(self):
        return self.prev_or_next("i_next_word_3=", 0, 3)
        
    def i_next_pos(self):
        return self.prev_or_next("i_next_pos=", 0, 1, True)
    
    def i_next_pos_2(self):
        return self.prev_or_next("i_next_pos_2=", 0, 2, True)
        
    def i_next_pos_3(self):
        return self.prev_or_next("i_next_pos_3=", 0, 3, True)
        
    def j_next_word(self):
        return self.prev_or_next("j_next_word=", 1, 1)
    
    def j_next_word_2(self):
        return self.prev_or_next("j_next_word_2=", 1, 2)
        
    def j_next_word_3(self):
        return self.prev_or_next("j_next_word_3=", 1, 3)
        
    def j_next_pos(self):
        return self.prev_or_next("j_next_pos=", 1, 1, True)
    
    def j_next_pos_2(self):
        return self.prev_or_next("j_next_pos_2=", 1, 2, True)
        
    def j_next_pos_3(self):
        return self.prev_or_next("j_next_pos_3=", 1, 3, True)

    def words_between(self):
        """return all words between two entities as a list"""
        name = "w_betw="
        values = []
        cur_filename = None
        r = None
        for pair in self.pairs:
            if not pair[3]:
                values.append(name + self.F)
            else:
                i_end = pair[0][4][1]
                j_start = pair[1][4][0]
                filename = pair[4]
                if cur_filename != filename:
                    r = document_reader.RelExtrReader(filename)
                    cur_filename = filename
                values.append(name + str(r.get_words(pair[0][3], i_end, j_start)))
        return values

    def first_word_between(self):
        """return first word between two entities as a list"""
        name = "fw_betw="
        values = []
        cur_filename = None
        r = None
        for pair in self.pairs:
            if not pair[3]:
                values.append(name + self.F)
            else:
                i_end = pair[0][4][1]
                j_start = pair[1][4][0]
                if j_start - i_end < 3:
                    values.append(name + self.F)
                else:
                    filename = pair[4]
                    if cur_filename != filename:
                        r = document_reader.RelExtrReader(filename)
                        cur_filename = filename
                    values.append(name + r.get_words(pair[0][3], i_end, j_start)[0])
        return values

    def last_word_between(self):
        """return last word between two entities as a list"""
        name = "lw_betw="
        values = []
        cur_filename = None
        r = None
        for pair in self.pairs:
            if not pair[3]:
                values.append(name + self.F)
            else:
                i_end = pair[0][4][1]
                j_start = pair[1][4][0]
                if j_start - i_end < 3:
                    values.append(name + self.F)
                else:
                    filename = pair[4]
                    if cur_filename != filename:
                        r = document_reader.RelExtrReader(filename)
                        cur_filename = filename
                    values.append(name+r.get_words(pair[0][3], i_end, j_start)[-1])
        return values

    def no_words_between(self):
        """Returns true if there are no words between two mentions"""
        name = "no_w_betw="
        values = []
        
        for pair in self.pairs:
            i_start, i_end = pair[0][4]
            j_start, j_end = pair[1][4]
            if (i_end == j_start or j_end == i_start) and pair[3]:
                values.append(name + self.T)
            else:
                values.append(name + self.F)
        return values

    def take_svm_tk_results(self):
        """
        feed instances through 43 SVM classifier,
        get their results and pick the most confident label, return as feature value
        """
        name = "svm_classification="
        values = []
        labels = svmlight_wrapper.RELATIONS
        num_rels = len(labels)
        # 2d array to store svm results: rows - label, cols - instances
        results = np.zeros([num_rels, len(self.pairs)])
        for num, l in enumerate(labels):
            svm_feed_filename = os.path.join(svmlight_wrapper.SVM_DATA_PATH,
                                             "svm_{}_{}".format(self.svm_prefix, l))
            classified = svmlight_wrapper.classify(l, svm_feed_filename)
            if classified is None:
                classified = [-1] * len(self.pairs)
            results[num] = classified
        # now get through each column, find the most confident
        for i in range(len(self.pairs)):
            instance = results[:, i]
            best_label = np.max(instance)
            if best_label > 0:
                values.append(name + labels[list(instance).index(best_label)])
            else:
                values.append(name + "no_rel")
        return values


class CoreferenceResolver(object):
    """
    CoreferenceResolver class is a classifier to detect named entities
    using TaggerFrame as feature extractor and Mallet Maxent as its algorithm
    """

    def __init__(self):
        super(CoreferenceResolver, self).__init__()
        os.chdir(PROJECT_PATH)
        self.ft = FeatureTagger()
        self.trainfile \
            = os.path.join('result', 'trainFeatureVector.txt')
        self.targetfile \
            = os.path.join('result', 'targetFeatureVector.txt')
        self.windows = sys.platform.startswith("win")
        if self.windows:
            self.me_script \
                = os.path.join(".", "mallet-maxent-classifier.bat")
        else:
            self.me_script \
                = os.path.join(".", "mallet-maxent-classifier.sh")
        self.modelfile = os.path.join("result", "model")

    def train(self, train_filename):
        """train MaxEnt module with a given data file"""
        self.ft.read_input_data(train_filename)
        self.ft.feature_matrix(self.trainfile)
        subprocess.check_call(
            [self.me_script, "-train",
             "-model=" + self.modelfile,
             "-gold=" + self.trainfile])

    def classify(self, target_filename):
        """Run MaxEnt classifier to classify target file"""
        self.ft.read_input_data(target_filename, labeled=False)
        self.ft.feature_matrix(self.targetfile, train=False)
        if not os.path.isfile(self.modelfile):
            raise Exception("Model not found.")
        resultfile = os.path.join("result", "result.txt")

        maxent_proc = subprocess.Popen(
            [self.me_script, "-classify",
             "-model=" + self.modelfile,
             "-input=" + self.targetfile],
            stdout=subprocess.PIPE)
        with open(resultfile, "w") as outf:
            for line in maxent_proc.communicate()[0]:
                outf.write(line)

        # evaluate the result
        target_name = target_filename.split("/")[-1].split(".")[0]
        subprocess.check_call(
            ["python", "relation-evaluator.py",
             "data/%s.gold" % target_name, resultfile])

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        help="name of train set file",
        default=os.path.join(PROJECT_PATH, 'data', 'rel-trainset.gold')
    )
    parser.add_argument(
        "-t",
        help=
        "name of target file, if not given, program will ask users after training",
        default=None
    )
    start_time = time.time()
    args = parser.parse_args()

    cor = CoreferenceResolver()
    cor.train(args.i)
    if args.t is None:
        try:
            target = input(
                "enter a test file name with its path\n"
                + "(relative or full, default: data/rel-devset.raw): ")
        # if input is empty
        except SyntaxError:
            target = "rel-testset.raw"
    else:
        target = args.t
    cor.classify(target)
    print "elapsed: " + str(time.time() - start_time)

