## About
 * **User**: [pltrdy](https://github.com/pltrdy)
 * **Date**: 2017-07-19
 * **Commit**: [update readme a3b7a61](https://github.com/pltrdy/tf_rnnlm/commit/a3b7a61a4ad360d3e45d7bf9c52ed3016798729e)
 * **Dataset**: PTB
 * **Hardware**: GTX 1080

## small

```json
{
  "num_samples": 1024,
  "lr_decay": 0.5,
  "max_epoch": 4,
  "max_grad_norm": 5,
  "epoch": 14,
  "init_scale": 0.1,
  "fast_test": false,
  "cell": "lstm",
  "step": 0,
  "hidden_size": 200,
  "vocab_size": 10000,
  "num_layers": 2,
  "num_steps": 0,
  "learning_rate": 1.0,
  "max_max_epoch": 13,
  "batch_size": 64,
  "keep_prob": 1.0
}
```

## medium

```json
{
  "num_steps": 0,
  "step": 0,
  "init_scale": 0.05,
  "keep_prob": 0.5,
  "learning_rate": 1.0,
  "max_max_epoch": 39,
  "epoch": 40,
  "fast_test": false,
  "max_epoch": 6,
  "num_layers": 2,
  "hidden_size": 650,
  "batch_size": 64,
  "vocab_size": 10000,
  "cell": "lstm",
  "num_samples": 1024,
  "lr_decay": 0.8,
  "max_grad_norm": 5
}
```

## large

```json
{
  "learning_rate": 1.0,
  "init_scale": 0.04,
  "fast_test": false,
  "step": 0,
  "max_max_epoch": 55,
  "vocab_size": 10000,
  "num_steps": 0,
  "lr_decay": 0.8695652173913044,
  "batch_size": 64,
  "num_samples": 1024,
  "max_epoch": 14,
  "cell": "lstm",
  "num_layers": 2,
  "max_grad_norm": 10,
  "epoch": 56,
  "keep_prob": 0.35,
  "hidden_size": 1500
}
```

## Results
|config|train|valid|test|wps|time|
|---|---|---|---|---|---|
|small|39.031|127.949|124.292|94838|3m9.691s|
|medium|33.130|102.652|99.381|29418|21m12.092s|
|large|21.122|95.310|90.658|7057|122m48.261s|