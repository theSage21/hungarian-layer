import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment


class Hungarian(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, seq1, seq2):
        # B, S1, D
        # B, S2, D
        R1 = []
        for s1, s2 in zip(torch.unbind(seq1, 0),
                          torch.unbind(seq2, 0)):
            cost = []
            for i in s1:
                row = []
                for j in s2:
                    d = torch.squeeze(torch.norm((i - j), 2))
                    row.append(d)
                row = torch.squeeze(torch.stack(row))
                cost.append(row)
            cost = torch.stack(cost, 0).numpy()
            row_ind, col_ind = linear_sum_assignment(cost)
            row_remain = set(range(cost.shape[0])) - set(row_ind)
            col_remain = set(range(cost.shape[1])) - set(col_ind)
            row_ind = list(row_ind) + list(sorted(row_remain))
            col_ind = list(col_ind) + list(sorted(col_remain))
            row = torch.stack([s1[i, :] for i in row_ind], 0)
            R1.append(row)

        reorder = torch.stack(R1, 0)
        return reorder


if __name__ == '__main__':
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

    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    hung = Hungarian()
    a = hung(x, y)
    print(a.shape)
    print(a)
