#!/bin/bash

MODIFIED_FILES="$1"

RST_FILES=""
for F in $MODIFIED_FILES
do
    if [[ $F == *.rst ]]
    then
        RST_FILES="$RST_FILES $F"
    fi
done
echo "List of Changed RST Files: $RST_FILES"
if [ -z "$RST_FILES" ]; then
    echo "No RST Files have changed -- nothing to do in this PR"
else
    pwd
    RST_FILES="$RST_FILES source/rst/index_toc.rst"
    make website FILES="$RST_FILES" THEMEPATH="theme/lectures-python-programming.theme"
    ls _build/website/jupyter_html/*
fi