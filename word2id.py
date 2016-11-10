""" Generate 'word_to_id' file (saved as OUTPUT_FILE) from a document (INPUT_FILE) """
import reader
import sys
import pickle

if len(sys.argv) < 3:
	print("[ERROR] Two required parameter: word2id.py [INPUT_FILE] [OUTPUT_FILE]")
	exit(1)


in_path = sys.argv[1]
out_path = sys.argv[2]

word_to_id = reader._build_vocab(in_path)
with open(out_path, 'w') as f:
	pickle.dump(word_to_id, f)
