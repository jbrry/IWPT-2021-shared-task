#!/bin/bash

cd data

echo "Downloading UD data..."$'\n'

curl --remote-name-all https://lindat.cz/repository/xmlui/bitstream/handle/11234/1-3424{/ud-treebanks-v2.7.tgz}

tar -xvzf ud-treebanks-v2.7.tgz
rm ud-treebanks-v2.7.tgz
echo $'\n'"Done"
