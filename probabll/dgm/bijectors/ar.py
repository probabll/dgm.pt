def ar_modules(
    units,
    hidden_layers,
    hidden_activation,
    output_activation=torch.nn.Identity,
    hidden_diagonal_zeros=False,
    output_diagonal_zeros=False,
):
    """
    A FFNN using autoregressive linear layers.
    :param units: D
    :param hidden_layers: number of hidden layers
    :param hidden_activation: activation between linear layers
    :param output_activation: output activation
    :param hidden_diagonal_zeros: should hidden layers mask the diagonal
    :param output_diagonal_zeros: should the output layer mask the diagonal
    :return: a list of torch modules
    """
    modules = []
    for _ in range(hidden_layers):
        modules.append(
            AutoregressiveLinear(units, diagonal_zeros=hidden_diagonal_zeros)
        )
        modules.append(hidden_activation())
    modules.append(AutoregressiveLinear(units, diagonal_zeros=output_diagonal_zeros))
    modules.append(output_activation())
    return modules


def ff_modules(
    input_size,
    output_size,
    hidden_size,
    hidden_layers,
    hidden_activation,
    output_activation=torch.nn.Identity,
):
    """
    A FFNN.
    :param input_size:
    :param output_size:
    :param hidden_size:
    :param hidden_layers:
    :param hidden_activation:
    :param output_activation:
    :return: a list of torch modules
    """
    modules = []
    units = input_size
    for _ in range(hidden_layers):
        modules.append(nn.Linear(units, hidden_size))
        modules.append(hidden_activation())
        units = hidden_size
    modules.append(nn.Linear(units, output_size))
    modules.append(output_activation())
    return modules
