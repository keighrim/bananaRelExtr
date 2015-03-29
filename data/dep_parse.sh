#! /bin/bash

rm depparsed-files/*
cd rawtext-files
for f in *; do
    ../../lib/stanford-parser/lexparser_uncollapsed.sh \
        "$f" > ../depparsed-files/"$f".depparse
    echo "$f"
done

