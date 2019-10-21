# Training Log tracker


## Optimizer

```json
{
"optimizer":[
  {
      "name":"SGD",
      "lr":0.001,
      "momentum":0.9,
      "dampening":0,
      "weight_decay":1e-5,
      "nesterov":false
    },{
      "name":"Adadelta",
      "r":1.0,
      "rho":0.9,
      "eps":1e-06,
      "weight_decay":0
    },{
      "name":"Adagrad",
      "lr":0.01,
      "lr_decay":0,
      "weight_decay":0,
      "initial_accumulator_value":0
    },{
      "name":"Adam",
      "lr":0.001,
      "betas":[0.9,0.999],
      "eps":1e-08,
      "weight_decay":0,
      "amsgrad":false
    },{
      "name":"RMSprop",
      "lr":0.01,
      "alpha":0.99,
      "eps":1e-08,
      "weight_decay":0,
      "momentum":0,
      "centered":false
    },{
      "name":"Rprop",
      "lr":0.01,
      "betas":[0.5, 1.2],
      "step_sizes":[1e-06, 50]
   }]
 }
```

## Training Params
```json
{
  "epochs":1000,
  "batch_size":64,
  "dataset":"",
  "checkpoint":"",
  "log_dir":"",
  "optimizer":{
    "type":"SGD",
    "lr":0.001,
    "momentum":0.9,
    "dampening":0,
    "weight_decay":1e-5,
    "nesterov":false
  }
}
```
## Results

|model|dataset|bs|epochs|optimizer|lr|wright_decay|acc|mAP|model_size|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|model|dataset|bs|epochs|optimizer|lr|wright_decay|acc|mAP|model_size|