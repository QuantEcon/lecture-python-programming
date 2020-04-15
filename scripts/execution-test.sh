echo "PR: $TRAVIS_PULL_REQUEST"
echo "COMMIT RANGE: $TRAVIS_COMMIT_RANGE"
CHANGED_FILES=$(git diff --name-only $TRAVIS_COMMIT_RANGE | grep '\.rst' | tr '\n' ' ')
#Check for Full Deletions
SPHINX_FILES=""
for f in $CHANGED_FILES
do
    if [ -f $f ]
    then
        SPHINX_FILES="$SPHINX_FILES $f"
    fi
done
echo "List of Changed Files: $SPHINX_FILES"
if [ -z "$SPHINX_FILES" ]; then
    echo "No RST Files have changed -- nothing to do in this PR"
else
    make coverage FILES="$SPHINX_FILES"
fi