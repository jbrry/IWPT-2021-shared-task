#!/bin/bash

# 2020 data (should also be available on Lindat): http://ufal.mff.cuni.cz/~zeman/soubory/iwpt2020-train-dev.tgz

cd data

mkdir -p iwpt2021stdata
cd iwpt2021stdata

curl --remote-name-all https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3728{/iwpt2021stdata.tgz}
tar -xf iwpt2021stdata.tgz
mkdir ud_treebanks_iwpt
mv UD_* ud_treebanks_iwpt/

echo "Done"