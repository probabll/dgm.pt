import sys
import argparse
import pprint
import pathlib
import numpy as np
import json
import math
from tqdm import tqdm
from collections import OrderedDict
import torch.nn.functional as F
from itertools import chain

import torch
from torch.distributions import Distribution, Bernoulli, Normal, Uniform
from torch.distributions.kl import kl_divergence

import dgm

from dgm.prior import PriorLayer
from dgm.conditional import ConditionalLayer, FFConditioner, Conv2DConditioner, TransposedConv2DConditioner, MADEConditioner
from dgm.likelihood import FullyFactorizedLikelihood, AutoregressiveLikelihood
from dgm.opt_utils import get_optimizer, ReduceLROnPlateau, load_model, save_model

from utils import load_mnist, Batcher


class InferenceModel(torch.nn.Module):    
    
    def __init__(self, x_size, z_size, y_size,
                 conditional_z: ConditionalLayer):
        super(InferenceModel, self).__init__()
        self.x_size = x_size
        self.z_size = z_size
        self.y_size = y_size
        self.conditional_z = conditional_z        
        
    def parameters(self):
        return self.conditional_z.parameters()
    
    def q_z(self, x, y):        
        h = torch.cat([x, y], -1) if self.y_size > 0 else x
        return self.conditional_z(h)
   
    def forward(self, x, noisy_x, y, gen_model, weight_kl, min_kl, step):                
        """VAE loss"""
        # Prior
        p_z = gen_model.p_z(x.size(0), x.device)
        # Posterior
        q_z = self.q_z(x, y)
        # [B, dz]
        z = q_z.rsample()
        # Likelihood
        p_x = gen_model.p_x(z, y=y, x=noisy_x)
        # [B]
        ll = p_x.log_prob(x).sum(-1)

        # [B]
        KL = kl_divergence(q_z, p_z).sum(-1) 
        ELBO = ll - KL
        loss = - (ll - torch.clamp(KL, min=min_kl) * weight_kl)

        display = OrderedDict()
        display['ELBO'] = ELBO.mean().item()
        display['KL'] = KL.mean().item()
        display['0s'] = (z == 0).float().mean(-1).mean().item()
        display['1s'] = (z == 1).float().mean(-1).mean().item()
               
        return loss.mean(), display

class GenerativeModel(torch.nn.Module):
    
    def __init__(self, x_size, z_size, y_size,
                 prior_z: PriorLayer,                  
                 conditional_x: ConditionalLayer):
        super(GenerativeModel, self).__init__()
        self.prior_z = prior_z  
        self.conditional_x = conditional_x        
        self.z_size = z_size
        self.x_size = x_size
        self.y_size = y_size
        
    def p_z(self, batch_size, device):
        return self.prior_z(batch_size, device)
    
    def p_x(self, z, y=None, x=None) -> Distribution:
        inputs = z if self.y_size == 0 else torch.cat([z, y], -1)            
        history = x if isinstance(self.conditional_x, AutoregressiveLikelihood) else None
        return self.conditional_x(inputs, history)
    
    def forward(self, x, y, inf_model, num_samples):   
        """Estimate likelihood of observation"""
        # Prior
        p_z = self.p_z(x.size(0), x.device)
        # Posterior                          
        q_z = inf_model.q_z(x, y)
        
        # Log-likelihood
        # []
        ll = []
        log_conditional = 0.
        ones = 0.
        zeros = 0.
        for s in range(num_samples):
            # [B, dz]
            z = q_z.rsample()
            # Likelihood
            p_x = self.p_x(z, y=y, x=x)
            log_conditional_x = p_x.log_prob(x).sum(-1)
            ll.append(
                # [1, B]                
                (log_conditional_x + p_z.log_prob(z).sum(-1) - q_z.log_prob(z).sum(-1)).unsqueeze(0)
            )
            # [B]
            log_conditional = log_conditional + log_conditional_x
            zeros = zeros + (z == 0).float().mean(-1).mean()
            ones = ones + (z == 1).float().mean(-1).mean()
        # [B]
        log_conditional = log_conditional / num_samples
        # [num_samples, B]
        ll = torch.cat(ll, 0)
        # [B]
        ll = ll.logsumexp(0) - math.log(num_samples)
        
        # Objective
        # [B]
        KL = kl_divergence(q_z, p_z).sum(-1)
        # [B]                    
        ELBO = log_conditional - KL            
           
        display = OrderedDict()
        display['ELBO'] = ELBO.mean().item()
        display['KL'] = KL.mean().item()
        display['LL'] = ll.mean().item()  
        display['0s'] = zeros.sum().item() / num_samples
        display['1s'] = ones.sum().item() / num_samples
        return ll.mean(), display
      

def config(**kwargs):
    """You can use kwargs to programmatically overwrite existing parameters"""

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Data
    parser.add_argument('--height', type=int, default=28,
        help='Used to specify the data_dim=height*width')
    parser.add_argument('--width', type=int, default=28,
        help='Used to specify the data_dim=height*width')
    parser.add_argument('--binarize', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=64)

    # Model and Architecture
    parser.add_argument('--latent_size', type=int, default=64,
        help="Dimensionality of latent code Z ~ N(0,I).")
    parser.add_argument('--conditional', default=False, action="store_true",
        help="Model P(x|y) and q(z|x,y)")
    parser.add_argument('--likelihood', type=str, default='bernoulli',
        help='Data likelihood',
        choices=['bernoulli']
    )    
    parser.add_argument('--prior', type=str, default="gaussian",
        choices=["gaussian"], help="Reserved for future use")
    parser.add_argument('--prior_params', type=str, default="0. 1.")
    parser.add_argument('--posterior', type=str, default="gaussian", 
        choices=["gaussian"], help="Reserved for future use")
    parser.add_argument('--hidden_sizes', type=str, default="500 500",
        help="Decoder's hidden layers.")
    parser.add_argument('--encoder', type=str, default="basic", choices=['basic', 'cnn'],
        help="Choose the encoder architecture.")    
    parser.add_argument('--decoder', type=str, default="basic", choices=['basic', 'cnn', 'made'],
        help="Choose the decoder architecture.")    

    # Optimization
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--early_stopping', type=int, default=10)
    parser.add_argument('--max_gradient_norm', type=float, default=1.)

    parser.add_argument('--gen_opt', type=str, default="adam")
    parser.add_argument('--gen_lr', type=float, default=1e-3)
    parser.add_argument('--gen_l2_weight', type=float, default=1e-4)
    parser.add_argument('--gen_momentum', type=int, default=0.)

    parser.add_argument('--inf_z_opt', type=str, default="adam")
    parser.add_argument('--inf_z_lr', type=float, default=1e-3)
    parser.add_argument('--inf_z_l2_weight', type=float, default=1e-4)
    parser.add_argument('--inf_z_momentum', type=int, default=0.)
    
    # Posterior collapse
    parser.add_argument('--kl_weight', type=float, default=1.)
    parser.add_argument('--kl_inc', type=float, default=0.)
    parser.add_argument('--min_kl', type=float, default=0.)
    parser.add_argument('--input_dropout', type=float, default=0.)

    # Metrics
    parser.add_argument('--ll_samples',
        type=int, default=10,
        help='Number of samples used to estimate likelihood of observations.')

    # Experiment
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./runs')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--logdir', type=str, default=None,
        help='Tensorboard logdir')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--validate', default=False, action="store_true",
        help="Validate a trained model (skip training)")
    parser.add_argument('--test', default=False, action="store_true",
        help="Test a trained model (skip training)")

    args, _ = parser.parse_known_args()
    # Convert strings to lists of floats/ints
    if isinstance(args.prior_params, str):
        args.prior_params = [float(v) for v in args.prior_params.split()]
    if isinstance(args.hidden_sizes, str):
        args.hidden_sizes = [int(v) for v in args.hidden_sizes.split()]    
    # overwrites
    for k, v in kwargs.items():
        args.__dict__[k] = v
    
    # Save hyperparameters
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)        
    
    # Log dir
    if args.logdir:
        args.logdir = pathlib.Path(args.logdir)
        args.logdir.mkdir(parents=True, exist_ok=True)    
    
    # reproducibility is good
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    return args


def config_from_file(filename):
    with open(filename) as f:
        hparams = json.load(f)
    return config(**hparams)


class Experiment:
    """
    Use this class to
    * load a dataset
    * build model and optimizer
    * get a batcher
    * train a model
    * load a trained model
    """
    
    def __init__(self, args):
        
        print("\n# Hyperparameters", file=sys.stderr)
        pprint.pprint(args.__dict__, stream=sys.stderr)

        print("\n# Data", file=sys.stderr)
        print(" - Standard MNIST", file=sys.stderr)    
        print(" - digit_dim=%d*%d" % (args.height, args.width), file=sys.stderr)
        print(" - data_dim=%d*%d" % (args.height, args.width), file=sys.stderr)
        train_loader, valid_loader, test_loader = load_mnist(
            args.batch_size, 
            save_to='{}/std/{}x{}'.format(args.data_dir, args.height, args.width),
            height=args.height, 
            width=args.width)
         
        x_size = args.height * args.width
        z_size = args.latent_size
        y_size = 10 if args.conditional else 0        
        
        # Configure prior
        if args.prior == 'gaussian':
            prior_type = Normal
        else:
            raise ValueError("Unknown prior: %s" % args.prior)
        p_z = PriorLayer(
            event_shape=z_size,
            dist_type=prior_type,
            params=args.prior_params
        )                    
        
        # Configure likelihood
        if args.likelihood == 'bernoulli':
            likelihood_type = Bernoulli
            decoder_outputs = 1 * x_size
        else:
            raise ValueError("Unknown likelihood: %s" % args.likelihood)
        
        if args.decoder == 'basic':
            likelihood_conditioner = FFConditioner(
                input_size=z_size + y_size, 
                output_size=decoder_outputs, 
                context_size=y_size, 
                hidden_sizes=args.hidden_sizes
            )            
        elif args.decoder == 'cnn':
            likelihood_conditioner = TransposedConv2DConditioner(
                input_size=z_size + y_size, 
                output_size=decoder_outputs,
                context_size=y_size, 
                input_channels=32, 
                output_channels=decoder_outputs // x_size, 
                last_kernel_size=7
            )            
        elif args.decoder == 'made':
            likelihood_conditioner = MADEConditioner(
                input_size=x_size + z_size + y_size, 
                output_size=decoder_outputs, 
                context_size=z_size + y_size,
                hidden_sizes=args.hidden_sizes,                 
                num_masks=1
            )
        else:
            raise ValueError("Unknown decoder: %s" % args.decoder)
        
        if args.decoder == 'made':
            conditional_x = AutoregressiveLikelihood(
                event_size=x_size, 
                dist_type=likelihood_type, 
                conditioner=likelihood_conditioner
            )
        else:
            conditional_x = FullyFactorizedLikelihood(
                event_size=x_size, 
                dist_type=likelihood_type, 
                conditioner=likelihood_conditioner
            )
         
        # CPU/CUDA device
        device = torch.device(args.device) 
        
        # Create generative model P(z)P(x|z)
        gen_model = GenerativeModel(
            x_size=x_size,
            z_size=z_size,             
            y_size=y_size,
            prior_z=p_z,
            conditional_x=conditional_x).to(device)
        print("\n# Generative Model", file=sys.stderr)
        print(gen_model, file=sys.stderr)
        
        # Configure posterior
        # Z|x,y
        if args.posterior == 'gaussian':
            encoder_outputs = z_size * 2
            posterior_type = Normal
        else:
            raise ValueError("Unknown posterior: %s" % args.posterior)            
        
        if args.encoder == 'basic':
            q_z = ConditionalLayer(
                event_size=z_size,
                dist_type=posterior_type,
                conditioner=FFConditioner(
                    input_size=x_size + y_size,
                    output_size=encoder_outputs, 
                    hidden_sizes=args.hidden_sizes                
                )
            )
        elif args.encoder == 'cnn':            
            q_z = ConditionalLayer(
                event_size=z_size,
                dist_type=posterior_type,
                conditioner=Conv2DConditioner(
                    input_size=x_size + y_size,
                    output_size=encoder_outputs,
                    context_size=y_size,
                    width=args.width,
                    height=args.height,
                    output_channels=256,
                    last_kernel_size=7
                )
            )
        else:
            raise ValueError("Unknown encoder architecture: %s" % args.encoder)
            
        inf_model = InferenceModel(
            x_size=x_size,
            z_size=z_size,             
            y_size=y_size,
            conditional_z=q_z
        ).to(device)
        print("\n# Inference Model", file=sys.stderr)
        print(inf_model, file=sys.stderr)

        print("\n# Optimizers", file=sys.stderr)
        gen_opt = get_optimizer(args.gen_opt, gen_model.parameters(), args.gen_lr, args.gen_l2_weight, args.gen_momentum)
        gen_scheduler = ReduceLROnPlateau(
            gen_opt, 
            factor=0.5, 
            patience=args.patience,
            early_stopping=args.early_stopping,
            mode='max', threshold_mode='abs')
        print(gen_opt, file=sys.stderr)   
        
        inf_z_opt = get_optimizer(args.inf_z_opt, inf_model.parameters(), 
                                  args.inf_z_lr, args.inf_z_l2_weight, args.inf_z_momentum)
        inf_z_scheduler = ReduceLROnPlateau(
            inf_z_opt, 
            factor=0.5, 
            patience=args.patience,
            mode='max', threshold_mode='abs')
        print(inf_z_opt, file=sys.stderr)
        
        self.optimizers = {'gen': gen_opt, 'inf_z': inf_z_opt}
        self.schedulers = {'gen': gen_scheduler, 'inf_z': inf_z_scheduler}
                        
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.models = {'gen': gen_model, 'inf': inf_model}        
        self.args = args
        
    def get_batcher(self, data_loader):
        batcher = Batcher(
            data_loader, 
            height=self.args.height, 
            width=self.args.width, 
            device=torch.device(self.args.device), 
            binarize=self.args.binarize, 
            onehot=True,
            num_classes=10
        )
        return batcher
    
    def load(self):
        load_model(self.models, self.optimizers, self.args.output_dir)()
    
    def save_config(self):
        with open("%s/hparams" % self.args.output_dir, "w") as f:
            json.dump(self.args.__dict__, f, sort_keys=True, indent=4)
            
    def train(self):

        self.save_config()
        print("\n# Training", file=sys.stderr)
        args = self.args
        gen_model, inf_model = self.models['gen'], self.models['inf']

        if args.logdir:
            from tensorboardX import SummaryWriter        
            writer = SummaryWriter(args.logdir)
        else:
            writer = None

        step = 1
        min_kl, weight_kl, kl_inc = args.min_kl, args.kl_weight, args.kl_inc

        for epoch in range(args.epochs):

            iterator = tqdm(self.get_batcher(self.train_loader))

            for i, (x_mb, y_mb) in enumerate(iterator):                        
                # [B, H*W]
                x_mb = x_mb.reshape(-1, args.height * args.width)
                # [B, 10]
                y_mb = y_mb.float()
                
                # Training mode and zero grad
                gen_model.train()
                inf_model.train()
                for opt in self.optimizers.values():
                    opt.zero_grad()
 
                # []
                noisy_x = x_mb
                loss, loss_dict = inf_model(x_mb, noisy_x, y_mb, gen_model, weight_kl, min_kl, step)                
                loss.backward()
                if args.max_gradient_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        parameters=chain(gen_model.parameters(), inf_model.parameters()),
                        max_norm=args.max_gradient_norm,
                        norm_type=float("inf"))
                for opt in self.optimizers.values():
                    opt.step()

                # TODO: use writer                
                iterator.set_postfix(OrderedDict((k, '{:4.2f}'.format(v)) for k, v in loss_dict.items()), refresh=False)
                step += 1

                # KL annealing
                weight_kl += kl_inc
                if weight_kl > 1.:
                    weight_kl = 1.

            val_ll, val_dict = self.validate()
            
            stop = self.schedulers['gen'].step(val_ll,
                callback_best=save_model(self.models, self.optimizers, args.output_dir),
                callback_reduce=load_model(self.models, self.optimizers, args.output_dir))
            for name, scheduler in self.schedulers.items():
                if name != 'gen':
                    scheduler.step(val_ll)

            if stop:
                print('Early stopping at epoch {:3}/{}'.format(epoch + 1, args.epochs))
                break

            print('Epoch {:3}/{} -- LL {:.2f} -- '.format(epoch + 1, args.epochs, val_ll) + \
                  ', '.join(['{}: {:4.2f}'.format(k, v) for k, v in sorted(val_dict.items())]))
        
        print("Loading best model...")
        self.load()
        print("Validation results")
        _, val_dict = self.validate()
        print('dev', ' '.join(['{}={:4.2f}'.format(k, v) for k, v in sorted(val_dict.items())]))

    def log_likelihood(self, batcher: Batcher, num_samples=10):
        """Check validation performance (note this will not update the learning rate scheduler"""
        args, gen_model, inf_model = self.args, self.models['gen'], self.models['inf']
        gen_model.eval()
        inf_model.eval()
        with torch.no_grad():
            total_ll = 0.            
            data_size = 0.
            return_dict = OrderedDict()
            for x_mb, y_mb in batcher:
                # [B, H*W]
                x_mb = x_mb.reshape(-1, args.height * args.width)
                # [B, 10]
                y_mb = y_mb.float()  
                # []
                ll, ll_dict = gen_model(x_mb, y_mb, inf_model, num_samples)                
                # turn mean into sum
                total_ll = total_ll + ll * x_mb.size(0)
                for k, v in ll_dict.items():
                    return_dict[k] = return_dict.get(k, 0.) + v * x_mb.size(0)
                data_size = data_size + x_mb.size(0)
            total_ll = total_ll / data_size
            for k, v in return_dict.items():
                return_dict[k] = v / data_size
        return total_ll, return_dict
    
    def validate(self):
        ll, return_dict = self.log_likelihood(
            self.get_batcher(self.valid_loader), 
            self.args.ll_samples, 
        )
        return ll, return_dict
    
    def test(self):
        """Check test performance (note this will not update the learning rate scheduler)"""
        ll, return_dict = self.log_likelihood(
            self.get_batcher(self.test_loader), 
            self.args.ll_samples, 
        )
        return ll, return_dict    
   
    
def main():
    args = config()
    if args.validate or args.test:
        print("Loading %s/hparams" % args.output_dir, file=sys.stderr)
        trained_args = config_from_file('%s/hparams' % args.output_dir)
        exp = Experiment(trained_args)
        print("Loading model...", file=sys.stderr)
        exp.load()
        if args.validate:
            _, val_dict = exp.validate()
            print('dev', ' '.join(['{}={:4.2f}'.format(k, v) for k, v in sorted(val_dict.items())]))
        if args.test:
            _, test_dict = exp.test()
            print('test', ' '.join(['{}={:4.2f}'.format(k, v) for k, v in sorted(test_dict.items())]))
    else:
        exp = Experiment(args)
        exp.train()

if __name__ == '__main__':
    main()
