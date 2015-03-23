# yago.py
# TJC
# Make dictionaries out of YAGO info for searching later

import os
import gc
import re
import cPickle

PROJECT_PATH = os.getcwd()

# I can't do this on project directory...
# too big for sharing through github or gDrive
YAGO_PATH = os.path.join(os.path.expanduser("~"), "yagoTransitiveType.tsv")
OUT_PATH = os.path.join(os.path.expanduser("~"), "yago_split")

entity_set = set()
count = 1

cur_entry = ""
cur_subtype = set()


def make_hash(string):
    """
    take a string and return its fist two letter
    this length-2 string will be used as an index to split YAGO file
    """
    hashed = ""
    for c in string:
        if len(hashed) > 1:
            return hashed
        else: 
            if c.isalpha():
                hashed += c
    while len(hashed) < 2:
        hashed += "_"
    return hashed

with open(YAGO_PATH, "r") as f, open("log.txt", "w") as log:
    print "loading complete"
    for line in f:
        #if count > 1000:
            #break
        name, _, subtypeOf = line.split()

        # skip 'owl:thing' line
        if not subtypeOf.startswith("<"):
            continue

        # trim and clean up name
        entry = name[1:-1].lower()
        # remove parenthesized explanation at the end of entry name
        if entry.endswith(")"):
            # log.write("paren: " + entry + "\n")
            entry = re.sub(r'_?\([^)]+\)$', '', entry)
        #try:
            #if re.search(entry, r'^\.+'):
                #print entry
                #log.write("dot..: " + entry + "\n")
                #entry = re.sub(r'^\.+', '', entry)
        #except:
            #print line
        if cur_entry == "":
            cur_entry = entry
        subtypeOf = subtypeOf[1:-1].lower()
        # clean subtype
        if subtypeOf.startswith("wordnet"):
            subtypeOf = "_".join(subtypeOf.split("_")[1:-1])
        elif subtypeOf.startswith("wikicate"):
            subtypeOf = "_".join(subtypeOf.split("_")[1:])

        # moving on to next item in yago database
        # we are assuming here that names in yago file are sorted
        if cur_entry != entry:
            count += 1
            filename = make_hash(cur_entry)
            with open(os.path.join(OUT_PATH, filename), "a") as split:
                #print cur_entry
                split.write("{}: {}\n".format(cur_entry, list(cur_subtype)))
            entity_set.add(cur_entry)
            cur_entry = entry
            cur_subtype = set()
        else:
            cur_subtype.add(subtypeOf)

# print len(entity_set)

# make a dict from each token of each YAGO entity to its full name for further lookup
entity_dict = {}
for entity in entity_set:
    if entity_dict.get(entity):
        entity_dict[entity].append(entity)
    else:
        entity_dict[entity] = [entity]
    for token in entity.split("_"):
        try:
            entity_dict[token].append(entity)
        except KeyError:
            entity_dict[token] = [entity]

print len(entity_dict.keys())

# finally pickle out indexed hash dict
with open('yago_entries.p', "wb") as picklejar:
    cPickle.dump(entity_dict, picklejar, protocol=cPickle.HIGHEST_PROTOCOL)
