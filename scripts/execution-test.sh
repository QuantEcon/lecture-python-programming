#!/bin/bash

CLEAN_BUILD=false
MODIFIED_FILES="$1"

RST_FILES=""
for F in $MODIFIED_FILES
do
    if [[ $F == environment.yml ]]
    then
        CLEAN_BUILD=true
        break
    fi
    #Extract List of RST Files
    if [[ $F == *.rst ]]
    then
        RST_FILES="$RST_FILES $F"
    fi
done

echo "List of Changed RST Files: $RST_FILES"
echo "Clean Build Requested: $CLEAN_BUILD"

if [ "$CLEAN_BUILD" = true ]
then
    echo "Running Clean Build"
    make coverage
elif [ -z "$RST_FILES" ]
then
    echo "No RST Files have changed -- nothing to do in this PR"
else
    RST_FILES="$RST_FILES source/rst/index_toc.rst"
    echo "Running Selecting Build with: $RST_FILES"
    make coverage FILES="$RST_FILES"
fi