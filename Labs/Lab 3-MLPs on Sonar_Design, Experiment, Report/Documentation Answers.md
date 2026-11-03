### 1. What does `nn.Sequential` accept as arguments, and in what order does it call the modules you give it?

`nn.Sequential` accepts module instances as positional arguments or an `OrderedDict` of named modules. It calls the modules in the order they are provided, passing each module's output as the input to the next module.

### 2. Given a model built with `nn.Sequential`, how do you get its first layer?

Use `model[0]` to access the first layer because indexing starts at zero. For a linear first layer, its weights and bias can be accessed using `model[0].weight` and `model[0].bias`.

### 3. Name two activation functions other than ReLU. What range of values does each output?

- **Sigmoid (`nn.Sigmoid`)**: Outputs values in `(0, 1)` for finite real inputs.
- **Tanh (`nn.Tanh`)**: Outputs values in `(-1, 1)` for finite real inputs.

### 4. For one optimizer you use: what arguments does it take, and what are their default values?

For `torch.optim.SGD`, the arguments and their default values are:

| Argument | Purpose | Default |
|---|---|---|
| `params` | Parameters or parameter groups to optimize | Required; no default |
| `lr` | Learning rate | `0.001` |
| `momentum` | Momentum factor | `0` |
| `dampening` | Dampening for momentum | `0` |
| `weight_decay` | L2 penalty coefficient | `0` |
| `nesterov` | Enables Nesterov momentum | `False` |
| `maximize` | Maximizes the objective instead of minimizing it | `False` |
| `foreach` | Controls use of the foreach implementation | `None` |
| `differentiable` | Enables autograd through the optimizer step | `False` |
| `fused` | Controls use of the fused implementation | `None` |

The baseline uses `torch.optim.SGD(model.parameters(), lr=0.01)`. This explicitly sets the learning rate to `0.01`, overriding the default of `0.001`.

### References

- [nn.Sequential](https://docs.pytorch.org/docs/stable/generated/torch.nn.Sequential.html)
- [nn.Sigmoid](https://docs.pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html)
- [nn.Tanh](https://docs.pytorch.org/docs/stable/generated/torch.nn.Tanh.html)
- [torch.optim.SGD](https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html)