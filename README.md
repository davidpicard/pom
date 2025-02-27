# PoM
Official implementation of the [Polynomial Mixer](https://arxiv.org/abs/2411.12663)

## Install

Create a local package and install it with the following commands:

```commandline

python setup.py sdist bdist_wheel
pip install .

```

## Usage

PoM is used to replace Multi-Head Attention. Its key parameters are the degree of the polynomials $d$ 
and the expansion factor $k$ of each polynomial. Assuming the original features have dimension $D$, then 
the internal state representation has dimension $dkD$. This has to be set according to your compute/memory 
budget, knowing that it is empirically better to have a higher $kD$ than a higher $d$.

The code to use a PoM layer is simple:

```python

from pom import PoM

pom = PoM(dimension, degree, expansion)

# residual self attention on token sequence X
X = X + pom(X)
# adding a residual feed-forward network as in transformers 
X = X + ffw(X)

```

### Causal inference

If you have a block causal mask, you can do iterative inference that has a constant memory cost. This is the 
case for example in video generation, where all the tokens of one frame depend only on the tokens of previous 
frames. This information can be encoded in a hidden state carried from one frame to another, as in the following 
example:

```python

# forward pass
state = [None for _ in range(n_layers)]
out = []
for f in range(n_frames):
    xf = x[:, f * n_tokens_p_frame:(f + 1) * n_tokens_p_frame, :]
    for l in range(n_layers):
        # self-attention
        x_sa, s = pom.state_forward(xf, xf, state=state)
        xf = xf + x_sa
        # ffw
        xf = xf + ffw(xf)
        state[l] = s
    out.append(xf)
x = torch.cat(out, dim=1)

```

## Citing us

```
@misc{picard24pom,
      title={PoM: Efficient Image and Video Generation with the Polynomial Mixer}, 
      author={David Picard and Nicolas Dufour},
      year={2024},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2411.12663} 
}
```