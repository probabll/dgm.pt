import torch
import torch.optim as optim
import torch.nn as nn
from functools import partial
from itertools import tee


def save_model(models: dict, optimizers: dict, args):
    def f():
        print('Saving model..')
        to_save = dict()
        for k, v in models.items():
            to_save['models.%s' % k] = v.state_dict()
        for k, v in optimizers.items():
            to_save['optimizers.%s' % k] = v.state_dict()
        torch.save(to_save, args.output_dir/'checkpoint.pt')
    return f


def load_model(models: dict, optimizers: dict, args):
    def f():
        print('Loading model..')
        checkpoint = torch.load(args.output_dir/'checkpoint.pt')
        for k, v in models.items():
            v.load_state_dict(checkpoint['models.%s' % k])
        for k, v in optimizers.items():
            v.load_state_dict(checkpoint['optimizers.%s' % k])
    return f


class RequiresGradSwitch:
    """
    Use this to temporarily switch gradient computation on/off for a subset of parameters:
    1. first you requires_grad(value), this will set requires_grad flags to a chosen value (False/True)
        while saving the original value of the flags
    2. then restore(), this will restore requires_grad to its original value
        i.e. whatever requires_grad was before you used requires_grad(value)
    """

    def __init__(self, param_generator):
        self.parameters = param_generator
        self.flags = None

    def requires_grad(self, requires_grad):
        if self.flags is not None:
            raise ValueError("Must restore first")
        self.parameters, parameters = tee(self.parameters, 2)
        flags = []
        for param in parameters:
            flags.append(param.requires_grad)
            param.requires_grad = requires_grad
        self.flags = flags

    def restore(self):
        if self.flags is None:
            raise ValueError("Nothing to restore")
        self.parameters, parameters = tee(self.parameters, 2)
        for param, flag in zip(parameters, self.flags):
            param.requires_grad = flag
        self.flags = None


def get_optimizer(name, parameters, lr, l2_weight, momentum=0.):
    if name is None or name == "adam":
        cls = optim.Adam
    elif name == "amsgrad":
        cls = partial(optim.Adam, amsgrad=True)
    elif name == "adagrad":
        cls = optim.Adagrad
    elif name == "adadelta":
        cls = optim.Adadelta
    elif name == "rmsprop":
        cls = partial(optim.RMSprop, momentum=momentum)
    elif name == 'sgd':
        cls = optim.SGD
    else:
        raise ValueError("Unknown optimizer: %s" % name)
    return cls(params=parameters, lr=lr, weight_decay=l2_weight)


class ReduceLROnPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    
    def __init__(self, *args, early_stopping=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.early_stopping = early_stopping
        self.early_stopping_counter = 0
    
    def step(self, metrics, epoch=None, callback_best=None, callback_reduce=None):
        current = metrics
        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
            self.early_stopping_counter = 0
            if callback_best is not None:
                callback_best()
        else:
            self.num_bad_epochs += 1
            self.early_stopping_counter += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0 # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            if callback_reduce is not None:
                callback_reduce()
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
            
        return self.early_stopping_counter == self.early_stopping

