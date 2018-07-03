#!/bin/bash -ex

rev=`git rev-parse --short HEAD`
d=`date +%Y-%m-%d-%H-%M-%S`

# convert arguements to string
# https://unix.stackexchange.com/questions/197792/joining-bash-arguments-into-single-string-with-spaces
old="$IFS"
IFS='_'
args="$*"
IFS=$old

dr="$d-$rev-$args"

mkdir $dr
cp Makefile $dr
cp ../build/bin/ligmg $dr
cp ../build/bin/ligmg_eigensolve $dr
cp graphs.txt $dr
flags="$*\\"$'\n'
sed -i.old "1s;^;LIGMG_FLAGS=$flags;" $dr/Makefile
echo "Created $dr"

make -C $dr
