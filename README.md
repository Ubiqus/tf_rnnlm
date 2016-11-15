# Recurrent Neural Network Language Model using TensorFlow

## Original Work

Our work is based on the [RNN LM tutorial on tensorflow.org](https://www.tensorflow.org/versions/r0.11/tutorials/recurrent/index.html#recurrent-neural-networks) following the paper from [Zaremba et al., 2014](https://arxiv.org/abs/1409.2329).

The tutorial uses the PTB dataset ([tgz](http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz)). We intend to work with various dataset, that's why we made names more generic by removing several "PTB" prefix initially present in the code.

We started to work from [master (commit 2d51a87f0727f8537b46048d8241aeebb6e48d6)](https://github.com/tensorflow/tensorflow/tree/e2d51a87f0727f8537b46048d8241aeebb6e48d6/tensorflow/models/rnn/ptb). (significant change since r0.11c). 

**See also:** [Nelken's "tf" repo](https://github.com/nelken/tf) inspired our work by the way it implements features we are interested in. 

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

# Train it. (it will automatically look for 'config' file in the model directory as no --config is set.
python word_lm.py --action train --data_path ./simple-examples/data --model_dir=./custom_model
```

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
