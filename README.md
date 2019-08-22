# dgm.pt

Pytorch code for building deep generative models

```bash
python setup.py develop
```

# Register parameterization

Use decorators to register a parameterization of a distribution, e.g. 

```python
@register_conditional_parameterization(Normal)
def make_normal(inputs, event_size):
    params = torch.split(inputs, event_size, -1)
    return Normal(loc=params[0], scale=F.softplus(params[1]))
```

Use conditioners to parameterize components. For example, here we show the prior, approximate posterior, and likelihood of the classic MNIST VAE:

```python
p_z = PriorLayer(
    event_shape=z_size,
    dist_type=Normal,
    params=[0., 1.]
)
q_z = ConditionalLayer(
    event_size=z_size,
    dist_type=Normal,
    conditioner=FFConditioner(
        input_size=x_size,
        output_size=z_size * 2,  # Gaussians take two parameters per unit
        hidden_sizes=[x_size // 2]
    )
)
p_x_given_z = FullyFactorizedLikelihood(
    event_size=x_size, 
    dist_type=Bernoulli, 
    conditioner=FFConditioner(
        input_size=z_size, 
        output_size=x_size * 1,   # Bernoullis take one parameter per unit
        hidden_sizes=[x_size // 2]
    )            
)
```
