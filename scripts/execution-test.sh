MODIFIED_FILES=$(<${HOME}/files_modified.json)
CHANGED_FILES=""
for F in $MODIFIED_FILES
do
    echo "$F"
    if [[ $F == *.rst ]]
    then
        CHANGED_FILES="$CHANGED_FILES $F"
    fi
done
echo "List of Changed Files: $CHANGED_FILES"
if [ -z "$CHANGED_FILES" ]; then
    echo "No RST Files have changed -- nothing to do in this PR"
else
    make coverage FILES="$CHANGED_FILES"
fi