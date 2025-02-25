# PoM
Official implementation of the [Polynomial Mixer](https://arxiv.org/abs/2411.12663)

## Install

Create a local package and install it with the following commands:

```

python setup.py sdist bdist_wheel
pip install .

```

## Usage

PoM is used to replace Multi-Head Attention. Its key parameters are the degree of the polynomials $d$ 
and the expansion factor $k$ of each polynomial. Assuming the original features have dimension $D$, then 
the internal state representation has dimension $dkD$. This has to be set according to your compute/memory 
budget, knowing that it is empirically better to have a higher $kD$ than a higher $d$.

The code to use a PoM layer is simple:

```

from pom import PoM

pom = PoM(dimension, degree, expansion)

# residual self attention on token sequence X
X = X + pom(X)
# adding a residual feed-forward network as in transformers 
X = X + ffw(X)

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