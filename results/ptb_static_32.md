## About
 * **User**: [pltrdy](https://github.com/pltrdy)
 * **Date**: 2017-07-18
 * **Commit**: [Using bach_size 32 dccd2c4](https://github.com/pltrdy/tf_rnnlm/commit/dccd2c42135e9db0b307fc79d0d1d63a90828c36)
 * **Dataset**: PTB dataset
 * **Hardware**: n/a

## small

```json
{
  "max_grad_norm": 5,
  "batch_size": 32,
  "epoch": 14,
  "num_layers": 2,
  "num_steps": 0,
  "max_max_epoch": 13,
  "hidden_size": 200,
  "lr_decay": 0.5,
  "learning_rate": 1.0,
  "init_scale": 0.1,
  "max_epoch": 4,
  "step": 0,
  "vocab_size": 10000,
  "num_samples": 1024,
  "cell": "lstm",
  "keep_prob": 1.0,
  "fast_test": false
}
```

## medium

```json
{
  "epoch": 40,
  "max_grad_norm": 5,
  "max_epoch": 6,
  "vocab_size": 10000,
  "keep_prob": 0.5,
  "num_steps": 0,
  "hidden_size": 650,
  "num_layers": 2,
  "max_max_epoch": 39,
  "fast_test": false,
  "num_samples": 1024,
  "learning_rate": 1.0,
  "batch_size": 32,
  "step": 0,
  "init_scale": 0.05,
  "cell": "lstm",
  "lr_decay": 0.8
}
```

## large

```json
{
  "batch_size": 32,
  "hidden_size": 1500,
  "cell": "lstm",
  "fast_test": false,
  "keep_prob": 0.35,
  "num_samples": 1024,
  "epoch": 56,
  "learning_rate": 1.0,
  "max_epoch": 14,
  "num_steps": 0,
  "max_grad_norm": 10,
  "max_max_epoch": 55,
  "step": 0,
  "num_layers": 2,
  "vocab_size": 10000,
  "init_scale": 0.04,
  "lr_decay": 0.8695652173913044
}
```

## Results
|config|train|valid|test|wps|time|
|---|---|---|---|---|---|
|small|27.913|123.896|119.496|42867|4m56.299s|
|medium|28.533|98.105|94.576|23214|26m51.753s|
|large|21.635|91.916|87.110|6185|140m55.675s|