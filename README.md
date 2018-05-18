Hungarian Layer
==============

Pytorch implementation of the Hungarian layer as described in <https://arxiv.org/abs/1712.02555>

Usage
-----


```python
import numpy as np

x = [[[0, 1], [0, 0], [1, 0]],
     [[0, 1], [0, 0], [1, 0]],
     [[0, 1], [0, 0], [1, 0]],
     [[0, 1], [0, 0], [1, 0]]
     ]
y = [[[0, 1], [0, 0]],
     [[0, 1], [0, 0]],
     [[0, 1], [0, 0]],
     [[0, 1], [0, 0]]]
x = np.array(x).astype(np.float64)
y = np.array(y).astype(np.float64)

x = torch.from_numpy(x)  # (4, 3, 2)
y = torch.from_numpy(y)  # (4, 2, 2)

hung = Hungarian()
a = hung(x, y)
print(a.shape)  #   (4, 3, 2)
print(a)
```
