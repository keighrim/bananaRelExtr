#! /bin/bash

rm synparsed/*
cd rawtext
for f in *; do
    ../../lib/stanford-parser/lexparser_phrasestructure.sh \
        "$f" > ../synparsed/"$f".psparse
    echo "$f"
done

