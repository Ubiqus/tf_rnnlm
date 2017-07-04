# Recurrent Neural Network Language Model using TensorFlow

## Content
* [Original Work](#orig)
* [Motivations](#motivations)
* [Quickstart](#quickstart)
* [Continue Training](#continue)
* [Getting a text perplexity with regard to the LM](#test)
* [Line by line loglikes](#loglikes)
* [Results on PTB Dataset](#ptb_results)


---

## [Original Work](#orig)

Our work is based on the [RNN LM tutorial on tensorflow.org](https://www.tensorflow.org/versions/r0.11/tutorials/recurrent/index.html#recurrent-neural-networks) following the paper from [Zaremba et al., 2014](https://arxiv.org/abs/1409.2329).

The tutorial uses the PTB dataset ([tgz](http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz)). We intend to work with various dataset, that's why we made names more generic by removing several "PTB" prefix initially present in the code.

**Original sources:** [TensorFlow v0.11 - PTB](https://github.com/tensorflow/tensorflow/tree/282823b877f173e6a33bbc9d4b9ad7dd8413ada6/tensorflow/models/rnn/ptb)

**See also:** 
- [Nelken's "tf" repo](https://github.com/nelken/tf) inspired our work by the way it implements features we are interested in. 
- [Benoit Favre's tf rnn lm](https://gitlab.lif.univ-mrs.fr/benoit.favre/tf_lm/blob/200645ab5aa446b72cf30c14355126062070f676/tf_lm.py)


## [Motivations](#motivations)
* Getting started with TensorFlow
* Make RNN LM manipulation easy in practice (easily look/edit configs, cancel/resume training, multiple outputs...)
* Train RNN LM for ASR using Kaldi (especially using `loglikes` mode)

## [Quickstart](#quickstart)

```shell
git clone https://github.com/pltrdy/tf_rnnlm
cd tf_rnnlm
./train.py --help
```

**Downloading PTB dataset:**
```shell
chmod +x tools/get_ptb.sh
./tools/get_ptb.sh
```

**Training small model:**
```shell
mkdir small_model
./train.py --data_path ./simple-examples/data --model_dir=./small_model --config small
```
**Training custom model:**
```shell
mkdir custom_model

# Generating new config file
chmod +x gen_config.py
./gen_config.py small custom_model/config

# Edit whatever you want
vi custom_model/config

# Train it. (it will automatically look for 'config' file in the model directory as no --config is set).
# It will look for 'train.txt', 'test.txt' and 'valid.txt' in --data_path
# These files must be present.
./train.py --data_path=./simple-examples/data --model_dir=./custom_model
```
**note:** data files are expected to be called `train.txt`, `test.txt` and `valid.txt`. Note that `get_ptb.sh` creates symlinks for that purpose

## Training all models and reporting results
```shell
./run.sh
```
Yes, that's all. It will train `small`, `medium` and `large` then generate a report like [this](results/template.md) using [report.sh](../tools/report.sh).   
**Feel free to share the report with us!**

## [Continue Training](#continue)
One can continue an interrupted training with the following command:
```shell
./train.py --data_path=./simple-examples/data --model_dir=./model
```
Where `./model` must contain `config`, `word_to_id`, `checkpoint` and the corresponding `.cktp` files.

## [Getting a text perplexity with regard to the LM](#test)
```shell
# Compute and outputs the perplexity of ./simple-examples/data/test.txt using LM in ./model
./test.py --data_path=./simple-examples/data --model_dir=./model
```

## [Line by line loglikes](#loglikes)
Running the model on each `stdin` line and returning its 'loglikes' (i.e. `-costs/log(10)`).

**Note:** in particular, it is meant to be used for Kaldi's rescoring.
```shell
cat ./data/test.txt | ./loglikes.py --model_dir=./model
```

## [Text generation](#generate)
Not documented yet
 

## [Results on PTB dataset](#ptb_results)
Configuration `small` `medium` and `large` are defined in `config.py` and are the same as in [tensorflow.models.rnn.ptb.ptb_word_lm.py:200](https://github.com/tensorflow/tensorflow/blob/e2d51a87f0727f8537b46048d8241aeebb6e48d6/tensorflow/models/rnn/ptb/ptb_word_lm.py#L200)


### Using `batch_size=32`
| config | train | valid  | test  |  speed   | training_time |
|--------|-------|--------|-------|----------|---------------|
| small  | 24.608| 118.848|113.366| ~49kWPS  |    4m17s      |
| medium | 26.068| 91.305 |87.152 | ~25kWPS  |    24m50s     |
| large  | 18.245| 84.603 |79.515 |  ~6kWPS  |    135m15s    |
|        |       |        |       |          |               |
| small  | 27.913|123.896 |119.496| ~42kWPS  |    4m56s      |
| medium | 28.533|98.105  |94.576 | ~23kWPS  |    26m51s     |
| large  | 21.635| 91.916 | 87.110|  ~6kWPS  |    140m675    |


### Using `batch_size=64`
| config | train | valid  | test  |  speed   | training_time |
|--------|-------|--------|-------|----------|---------------|
| small  | 32.202| 119.802|115.209| ~44kWPS  |    4m40s      |   
| medium | 31.591|  97.219| 93.450| ~24kWPS  |    25m0s      |   
| large  | 18.198|  88.675| 83.143|  ~9kWPS  |     95m25s    |   
|        |       |        |       |          |               |
| small  | 39.031| 127.949|124.292| ~94kWPS  |    3m9s       |   
| medium | 33.130| 102.652|99.381 | ~29kWPS  |    21m7s      |   
| large  | 21.122| 95.310 |90.658 |  ~7kWPS  |    112m48s    | 


**kWPS:** processing speed, i.e. thousands word per seconds.    
**Reported time** are `real` times (see [What do 'real', 'user' and 'sys' mean in the output of time(1)?](http://stackoverflow.com/a/556411/5903959)   
**Testing** is done using softmax on transposed weights. ([docs/transpose.md](docs/dynamic.md))    
**For faster results** increasing `batch_size` should speed up the process, with a small perplexity increase as a side effect and an increased GPU Memory consumption. (which can fire Out Of Memory exception)

## Contributing
Please do!   
**Fork** the repo -> **edit** the code -> **commit** with descriptive commit message -> open a **pull request**   
You can also open an issue for any discussion about bugs, performance or results.   
Please also share your results with us! (see [sharing your results](docs/share_results.md))
