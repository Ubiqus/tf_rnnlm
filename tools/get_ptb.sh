#!/bin/bash

echo "Downloading PTB Dataset... (http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz)"
wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

echo "Extracting..."
tar -xzf simple-examples.tgz

# Creating symlinks because PTB use 'ptb.' prefixed files. Our script wants files without this prefix
cd simple-examples/data
for f in ptb.*; do ln -s "$f"  "$(echo "$f" | cut -c 5-)"; done;
cd ../..
