#!/bin/bash

echo "Downloading PTB Dataset... (http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz)"
wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

# wget may produce simple-examples.tgz.1 if the file already exists
# we extract the last one
file=$(ls -srl | awk '{print $NF}' | grep simple-examples.tgz | head -1)

echo "Extracting $file"
tar -xzf "$file"

# Creating symlinks because PTB use 'ptb.' prefixed files. Our script wants files without this prefix
cd simple-examples/data
for f in ptb.*; do ln -s "$f"  "$(echo "$f" | cut -c 5-)"; done;
cd ../..

echo "Creating symlink ./data -> simple-examples/data"
ln -s simple-examples/data data
