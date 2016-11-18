# Recurrent Neural Network Language Model using TensorFlow

## Original Work

Our work is based on the [RNN LM tutorial on tensorflow.org](https://www.tensorflow.org/versions/r0.11/tutorials/recurrent/index.html#recurrent-neural-networks) following the paper from [Zaremba et al., 2014](https://arxiv.org/abs/1409.2329).

The tutorial uses the PTB dataset ([tgz](http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz)). We intend to work with various dataset, that's why we made names more generic by removing several "PTB" prefix initially present in the code.

**Original sources:** [TensorFlow v0.11 - PTB](https://github.com/tensorflow/tensorflow/tree/282823b877f173e6a33bbc9d4b9ad7dd8413ada6/tensorflow/models/rnn/ptb)

**See also:** 
- [Nelken's "tf" repo](https://github.com/nelken/tf) inspired our work by the way it implements features we are interested in. 
- [Benoit Favre's tf rnn lm](https://gitlab.lif.univ-mrs.fr/benoit.favre/tf_lm/blob/200645ab5aa446b72cf30c14355126062070f676/tf_lm.py)

## Quickstart

```
git clone https://github.com/pltrdy/tf_rnnlm
cd tf_rnnlm
python word_lm.py --help
```

**Note** that `word_lm.py` imports `reader.py` not `tensorflow.models.rnn.ptb` (i.e. it does not use your tensorflow installation). It is therefore compatible with tensorflow r0.11c as long as you don't use `word_lm.py` alone. 

Downloading PTB dataset: 
```
chmod +x get_ptb.sh
./get_ptb.sh
```

Training small model:
```
mkdir small_model
python word_lm.py --action train --data_path ./simple-examples/data --model_dir=./small_model --config small
```
Training custom model:
```
mkdir custom_model

# Generating new config file
chmod +x gen_config.py
./gen_config.py small custom_model/config

# Edit whatever you want
vi custom_model/config

# Train it. (it will automatically look for 'config' file in the model directory as no --config is set).
# It will look for 'train.txt', 'test.txt' and 'valid.txt' in --data_path
# These files must be present.
python word_lm.py --action train --data_path=./simple-examples/data --model_dir=./custom_model
```
**note:** data files are expected to be called `train.txt`, `test.txt` and `valid.txt`. Note that `get_ptb.sh` creates symlinks for that purpose

## Continue Training (--action continue) 
One can continue an interrupted training with the following command:
```
python word_lm.py --action continue --data_path=./simple-examples/data --model_dir=./model
```
Where `./model` must contain `config`, `word_to_id`, `checkpoint` and the corresponding `.cktp` files.

## Getting a text perplexity with regard to the LM (--action test)
```
# Compute and outputs the perplexity of ./simple-examples/data/test.txt using LM in ./model
python word_lm.py --action test --data_path=./simple-examples/data --model_dir=./model
```

## Line by line perplexity (--action ppl)
Running the model on each `stdin` line and returning its perplexity
```
cat ./data/test.txt | python word_lm.py --action ppl --model_dir=./model
```

## Line by line prediction (--action predict)
Running the model on each `stdin` line and prints a prediction informations
```
cat ./data/test.txt | python word_lm.py --action ppl --model_dir=./model
```
It will print a json object with the following structure:
+ ppl: perplexity (float)
+ predictions: (array, for each word of the line)
  + word: current word, 
  + target: next word, 
  + prob: probability associated with the target, 
  + pred_word: predicted word
  + pred_prob: probability associated with the predicted word


```

Good language models will tend to minimize `| prob - pred_prob | `

## Results

Quoting [tensorflow.models.rnn.ptb.ptb_word_lm.py:22](https://github.com/tensorflow/tensorflow/blob/e2d51a87f0727f8537b46048d8241aeebb6e48d6/tensorflow/models/rnn/ptb/ptb_word_lm.py#L22):
> There are 3 supported model configurations:
> 
> | config | epochs | train | valid  | test  |
> |--------|--------|-------|--------|-------|
> | small  | 13     | 37.99 | 121.39 | 115.91|
> | medium | 39     | 48.45 |  86.16 |  82.07|
> | large  | 55     | 37.87 |  82.62 |  78.29|
> The exact results may vary depending on the random initialization.
