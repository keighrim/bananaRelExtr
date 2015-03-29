# /usr/bin/python
# -*- coding: utf-8 -*-

"""
This program contains:
File reader for main script (raw text, syntactic parses, dependency parses)
as well as meny helper function for tree features and WordNet features
(Most code here is from last assignment for coreference resolution)
"""
import collections
import os
import pprint
import re
import sys
import nltk

from nltk.corpus import wordnet as wn
from nltk.tree import ParentedTree as ptree


reload(sys)
sys.setdefaultencoding('utf8')

__author__ = ["Keigh Rim", "Todd Curcuru", "Yalin Liu"]
__date__ = "3/20/2015"
__email__ = ['krim@brandeis.edu', 'tcurcuru@brandeis.edu', 'yalin@brandeis.edu']

PROJECT_PATH = os.getcwd()
DATA_PATH = os.path.join(PROJECT_PATH, "data")
POS_DATA_PATH = os.path.join(DATA_PATH, "postagged-files")
RAW_DATA_PATH = os.path.join(DATA_PATH, "rawtext-files")
DEPPARSE_DATA_PATH = os.path.join(DATA_PATH, "depparsed-files")
SYNPARSE_DATA_PATH = os.path.join(DATA_PATH, "parsed-files")

class RelExtrReader(object):
    """
    reader is the main class for reading document files in project dataset
    Basically a reader object represent a document in corpus
    """
    def __init__(self, filename):
        super(RelExtrReader, self).__init__()
        self.filename = filename
        # POS tagged
        self.tokenized_sents = self.tokenize_sent()
        # Dep parse trees
        self.depparse_trees = self.load_dep_parse()
        # PS parse trees
        self.synparsed_trees = self.load_syn_parse()

    def tokenize_sent(self):
        """separate out words and POS tags from .postagged files"""
        sents = []
        with open(os.path.join(POS_DATA_PATH,
                               self.filename + ".raw.tag")) as document:
            for line in document:
                # need to be careful about underscored characters
                line = re.sub(r"\b(_+)_([^_])", r"-_\2", line)
                if line != "\n":
                    sents.append(self.tokenize(line))
        return sents

    def get_all_sents(self):
        sents = []
        for sent in self.tokenized_sents:
            sents.append([w for w, _ in sent])
        return sents

    def write_raw_sents(self):
        """write a cleaned out raw text file from a given .postagged file"""
        if not os.path.exists(RAW_DATA_PATH):
            os.makedirs(RAW_DATA_PATH)
        with open(os.path.join(RAW_DATA_PATH, self.filename + ".raw"), "w") as outf:
            for sentence in self.get_all_sents():
                sent = " ".join(sentence)
                sent = sent.replace("(", "[")
                sent = sent.replace(")", "]")
                outf.write(sent)
                outf.write("\n")

    def load_syn_parse(self):
        """load a string of syntactic parse into NLTK ParentedTree structure"""
        sents = []
        with open(os.path.join(SYNPARSE_DATA_PATH,
                               self.filename + ".raw.psparse")) as parse:
            sent = ""
            for line in parse:
                if line == "\n":
                    try:
                        sents.append(ptree.fromstring(sent))
                        sent = ""
                    except ValueError:
                        print "TREE_LOAD_ERROR_AT: ", self.filename, sent
                        raise()
                else:
                    sent += line.strip()
        return sents

    def load_dep_parse(self):
        """load a string of Stanford Dependency parse into a data structure"""
        trees = []
        sent = {}
        with open(os.path.join(DEPPARSE_DATA_PATH,
                               self.filename + ".raw.depparse")) as parse:
            stanford_format_tree = ""
            for line in parse:
                if line == "\n":
                    # print "CONVERTING: ", stanford_format_tree, self.tokenized_sents[len(trees)]
                    # trees.append(
                    #     self.convert_to_nltk_format(stanford_format_tree,
                    #                                 self.tokenized_sents[len(trees)]))
                    # stanford_format_tree = ""
                    trees.append(sent)
                    sent = {}
                else:
                    # stanford_format_tree += line

                    # each line of Stanford dependency looks like this
                    # relation(govennor-gov_index, dependant-dep_index)
                    m = re.match(r"^(.+)\((.+)-([0-9']+), (.+)-([0-9']+)\)", line)
                    if m is None:
                        print "REGEX ERROR: ", line
                        continue
                    rel = m.groups()[0]
                    gov = m.groups()[1]
                    gov_idx = m.groups()[2]

                    # collapse primed nodes
                    if gov_idx.endswith("'"):
                        gov_idx = gov_idx.replace("'", "")
                    gov_idx = int(gov_idx) - 1
                    dep = m.groups()[3]
                    dep_idx = m.groups()[4]
                    if dep_idx.endswith("'"):
                        dep_idx = dep_idx.replace("'", "")
                    dep_idx = int(dep_idx) - 1
                    if gov_idx == dep_idx:
                        print gov, dep, "recursive relation"
                        continue

                    # final data structure will be a dict from
                    # token_index: (token_word,
                    #               dict of dependants of this token
                    #               (head_rel, head_idx, head_word))
                    # inner dict looks like
                    # relation: [ (idx, node_token) ]
                    try:
                        sent[gov_idx][1][rel].append((dep_idx, dep))
                    except KeyError:
                        sent[gov_idx] = (gov,                               # name
                                         collections.defaultdict(list),     # deps
                                         [])     # govs
                        sent[gov_idx][1][rel].append((dep_idx, dep))

                    try:
                        sent[dep_idx][2].extend([rel, gov_idx, gov])
                    except KeyError:
                        sent[dep_idx] = (dep,
                                         collections.defaultdict(list),
                                         [])
                        sent[dep_idx][2].extend([rel, gov_idx, gov])
        return trees

    @staticmethod
    def convert_to_nltk_format(stanford_tree, word_pos):
        nltk_string = ""
        for line in stanford_tree.split("\n"):
            if line == "\n" or len(line) == 0:
                continue
            m = re.match(r"^(.+)\((.+)-([0-9']+), (.+)-([0-9']+)\)", line)
            if m is None:
                print "REGEX ERROR: ", line
                continue
            rel = m.groups()[0].upper()
            gov = m.groups()[1]
            gov_idx = m.groups()[2]
            dep = m.groups()[3]
            dep_idx = m.groups()[4]

            # collapse primed nodes
            if gov_idx.endswith("'"):
                gov_idx = gov_idx.replace("'", "")
            gov_idx = int(gov_idx) - 1
            if dep_idx.endswith("'"):
                dep_idx = dep_idx.replace("'", "")
            dep_idx = int(dep_idx) - 1
            if gov_idx == dep_idx:
                print gov, dep, "recursive relation"
                continue

            nltk_string += "{0}\t{1}\t{2}\t{3}\n".format(
                dep, word_pos[dep_idx][1], str(gov_idx + 1), rel)
        # print "CONVERTED: ", nltk_string
        return nltk.dependencygraph.DependencyGraph(nltk_string)

    @staticmethod
    def tokenize(line):
        """returns [(word, pos)]"""
        tokens = []
        for token in line.split():
            token = token.split("_")
            if len(token) > 2:
                token = ["".join(token[:-1]), token[-1]]
            tokens.append(token)
        return tokens

    def get_tokens(self, sent, start, end):
        """get tokens at particular position of particular sentence"""
        if not isinstance(sent, int):
            sent = int(sent)
        if not isinstance(start, int):
            start = int(start)
        if not isinstance(end, int):
            end = int(end)
        return self.tokenized_sents[sent][start:end]

    def get_words(self, sent, start, end):
        """get words at particular position of particular sentence"""
        return [w for w, _ in self.get_tokens(sent, start, end)]

    def get_pos(self, sent, start, end):
        """get POS tags at particular position of particular sentence"""
        return [p for _, p in self.get_tokens(sent, start, end)]

    def get_dependency_tree(self, sent):
        """get dependency tree of particula sentnece"""
        return self.depparse_trees[sent]

    def compute_tree_path_length(self, sent, from_node, to_node):
        """
        compute distance between two nodes in NLTK tree,
        It's using NLTK methods, because we love NLTK
        """
        tree = self.synparsed_trees[sent]
        lowest_descendant = tree.treeposition_spanning_leaves(from_node, to_node)
        upward_path_length \
            = len(tree.leaf_treeposition(from_node)) - len(lowest_descendant)
        downward_path_length \
            = len(tree.leaf_treeposition(to_node)) - len(lowest_descendant)
        return upward_path_length, downward_path_length

    def get_dep_relation(self, sent, from_node, to_node):
        """get dependency relation of two nodes, if they are directly connected"""
        parse = self.depparse_trees[sent]
        # in case a mention is a relative pronoun go back to its antecedent
        while not parse.get(from_node):
            from_node -= 1
        while not parse.get(to_node):
            to_node -= 1

        # bi-directional check: i can govern j as well as i can be depending on j
        dependents = parse[from_node][1]
        for rel, dependent in dependents.iteritems():
            if dependent[0][0] == to_node:
                return rel
        governors = parse[from_node][2]
        if governors[1] == to_node:
            return governors[0]
        # if no relation found, return null
        return

    def get_dep_rel_path(self, sent, from_idx, to_idx):
        """get all dependency relations on the path between two targets"""
        dep_tree = self.depparse_trees[sent]

        def path_from_root(dep_tree, idx):
            """get path from node to root of a given dependency graph"""
            path = [idx]
            try:
                cur_node = dep_tree[idx]
            except KeyError:
                # this exception is caused by flaws in original data
                # but all of those problematic cases are no-rel
                # so, we'll just ignore them
                return path
            while cur_node[2][0] != "root":
                head_idx = cur_node[2][1]
                path.insert(0, head_idx)
                cur_node = dep_tree[head_idx]
            return path

        def oneway_relations_on_path(dep_tree, lower_node_idx, upper_node_idx):
            """return dependency relations on the oneway path from
            lower node to uppper node"""
            relations = []
            # if two indices are same, there is no path
            if lower_node_idx != upper_node_idx:
                cur_node = dep_tree[lower_node_idx]
                relations.append(cur_node[2][0])
                while cur_node[2][1] != upper_node_idx:
                    cur_node = dep_tree[cur_node[2][1]]
                    relations.append(cur_node[2][0])
            return relations

        path_to_from = path_from_root(dep_tree, from_idx)
        path_to_to = path_from_root(dep_tree, to_idx)

        lowest_ancestor = None
        for i in range(max(len(path_to_from), len(path_to_to))):
            try:
                if path_to_from[i] != path_to_to[i]:
                    lowest_ancestor = path_to_from[i-1]
                    break
            except IndexError:
                lowest_ancestor = path_to_from[i-1]
                break

        if lowest_ancestor is None:
            print "NO COMMON ANCESTOR"
            return None, None

        upward_relations \
            = oneway_relations_on_path(dep_tree, from_idx, lowest_ancestor)
        downward_relations \
            = oneway_relations_on_path(dep_tree, to_idx, lowest_ancestor)
        downward_relations.reverse()

        return upward_relations, downward_relations

    def is_subject(self, sent, token_offset):
        """return true if a token is playing subject"""
        parse = self.depparse_trees[sent]

        # since we are using collapsed dep parse trees,
        # in case a mention is a relative pronoun, it is collapsed in the parse
        # Thus, need to track backward for its antecedent
        while not parse.get(token_offset):
            token_offset -= 1

        governors = parse[token_offset][2]
        if governors[0] in ["nsubj", "nsubjpass", "subj"]:
            return True
        else:
            return False

    def is_object(self, sent, token_offset):
        """return true if a token is playing object"""
        parse = self.depparse_trees[sent]

        # for rel_pronuon
        while not parse.get(token_offset):
            token_offset -= 1
        governors = parse[token_offset][2]
        if governors[0] in ["dobj",  "iobj",  "obj"]:
            return True
        else:
            return False

    def get_deprel_verb(self, sent, noun_offset):
        """get a NE's governing verb is it's relation to its verb"""
        parse = self.depparse_trees[sent]

        # can this go into infinite loop? NO
        while True:
            # rel_pronoun
            while not parse.get(noun_offset):
                noun_offset -= 1
            # go through the list of governors
            governor = parse[noun_offset][2]
            if governor[0] in ("subj", "nsubj", "nsubjpass"):
                return "subj", governor[2]
            elif governor[0] in ("obj", "dobj", "iobj"):
                return "obj", governor[2]
                # if current token has a noun governor, move to that one,
                # continue to go up the tree
            elif governor[0] in ("appos", "nn", "poss", "conj_and", "conj_or"):
                new_noun_offset = governor[1]
                if new_noun_offset != noun_offset:
                    noun_offset = new_noun_offset
                else:
                    continue
            return (None, None)


    @staticmethod
    def in_same_synsets(verb_k, verb_t):
        """query WordNet to see two verbs are in same synset"""
        if verb_k is None or verb_t is None:
            return False
        else:
            verbset = set()
            for synset in wn.synsets(verb_k):
                verbset.update(synset.lemma_names())
            if verb_t in verbset:
                return True
            else:
                verbset = set()
                for synset in wn.synsets(verb_t):
                    verbset.update(synset.lemma_names())
                return verb_k in verbset


if __name__ == '__main__':
    # for filename in os.listdir(POS_DATA_PATH):
    #     filename = filename[:-8]
    #     r = reader(filename)
    #     r.write_raw_sents()

    r = RelExtrReader("APW20001001.2021.0521")
    # for tree in r.depparse_trees:
    #     for i, node in tree.iteritems():
    #         print len(node[2]), node[2]
    # print r.depparse_trees[3].get(-1)[2]
    # print r.tokenized_sents[3]
    # print r.depparse_trees[3]
    # print r.get_dep_rel_path(3, 0, 3)


    #print r.is_subject(15,4)
    #print r.is_object(15,4)
    #print r.get_dep_relation(17, 6, 10)
    #print r.in_same_synsets("said", "eat")

