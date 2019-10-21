import json
import torch.optim as optim
import datetime


def config_options(config_file):
  with open(config_file, 'r') as jf:
    cfg = json.load(jf)
  cfg["timestamp"] = '{0:%Y%m%d%H%M}'.format(datetime.datetime.now())
  return cfg


def config_optim(optim_cfg, params):

  if(optim_cfg["type"] == "SGD"):
    return optim.SGD(params,
                     lr=optim_cfg["lr"],
                     momentum=optim_cfg["momentum"],
                     dampening=optim_cfg["dampening"],
                     weight_decay=optim_cfg["weight_decay"],
                     nesterov=optim_cfg["nesterov"])
  elif(optim_cfg["type"] == "Adadelta"):
    return optim.Adadelta(params,
                          r=optim_cfg["r"],
                          rho=optim_cfg["rho"],
                          eps=optim_cfg["eps"],
                          weight_decay=optim_cfg["weight_decay"])
  elif(optim_cfg["type"] == "Adagrad"):
    return optim.Adagrad(
        params,
        lr=optim_cfg["lr"],
        lr_decay=optim_cfg["lr_decay"],
        weight_decay=optim_cfg["weight_decay"],
        initial_accumulator_value=optim_cfg["initial_accumulator_value"])
  elif(optim_cfg["type"] == "Adam"):
    return optim.Adam(params,
                      lr=optim_cfg["lr"],
                      betas=optim_cfg["betas"],
                      eps=optim_cfg["eps"],
                      weight_decay=optim_cfg["weight_decay"],
                      amsgrad=optim_cfg["amsgrad"])
  elif(optim_cfg["type"] == "RMSprop"):
    return optim.RMSprop(params,
                         lr=optim_cfg["lr"],
                         alpha=optim_cfg["alpha"],
                         eps=optim_cfg["eps"],
                         weight_decay=optim_cfg["weight_decay"],
                         momentum=optim_cfg["momentum"],
                         centered=optim_cfg["centered"])
  elif(optim_cfg["type"] == "Rprop"):
    return optim.Rprop(params,
                       lr=optim_cfg["lr"],
                       betas=optim_cfg["betas"],
                       step_sizes=optim_cfg["step_sizes"])
  else:
    raise ValueError("can't recognize the type of optimizer %s" %
                     optim_cfg["type"])
