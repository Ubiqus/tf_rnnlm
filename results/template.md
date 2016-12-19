**note** that such a result report can be generated using [tools/report.sh](../tools/report.sh)

## About
 * **User**: [<username>](https://github.com/<username>)
 * **Date**: YYYY-MM-DD
 * **Commit**: [<commit comment (1rst line)> <commit hash (first 7 chars)>](https://github.com/pltrdy/tf_rnnlm/commit/<commit hash>)
 * **Dataset**: <dataset description>
 * **Hardware**: <hardware description>

## small
*(show the different config (json) involved, examples below are `--config [small|medium|large] --batch_size 64`)*

```json
{
  "vocab_size": 10000, 
  "max_epoch": 4, 
  "keep_prob": 1.0, 
  "num_layers": 2, 
  "num_steps": 20, 
  "batch_size": 64, 
  "learning_rate": 1.0, 
  "epoch": 13, 
  "max_max_epoch": 13, 
  "max_grad_norm": 5, 
  "step": 0, 
  "hidden_size": 200, 
  "lr_decay": 0.5, 
  "init_scale": 0.1
}
```

## medium

```json
{
  "vocab_size": 10000, 
  "max_epoch": 6, 
  "keep_prob": 0.5, 
  "num_layers": 2, 
  "num_steps": 35, 
  "batch_size": 64, 
  "learning_rate": 1.0, 
  "epoch": 39, 
  "max_max_epoch": 39, 
  "max_grad_norm": 5, 
  "step": 0, 
  "hidden_size": 650, 
  "lr_decay": 0.8, 
  "init_scale": 0.05
}
```

## large

```json
{
  "vocab_size": 10000, 
  "max_epoch": 14, 
  "keep_prob": 0.35, 
  "num_layers": 2, 
  "num_steps": 35, 
  "batch_size": 64, 
  "learning_rate": 1.0, 
  "epoch": 55, 
  "max_max_epoch": 55, 
  "max_grad_norm": 10, 
  "step": 0, 
  "hidden_size": 1500, 
  "lr_decay": 0.8695652173913044, 
  "init_scale": 0.04
}
```

## Results 


|config|train|valid|test|wps|time|
|---|---|---|---|---|---|
|small||||||
|medium||||||
|large||||||

(format number like %.3f, time in min/sec like: *xx*m*yy.zzz*sec)
