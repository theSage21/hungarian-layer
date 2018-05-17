import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment


class Hungarian(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, seq1, seq2):
        # B, S1, D
        # B, S2, D
        R1, R2 = [], []
        for s1, s2 in zip(torch.unbind(seq1, 0),
                          torch.unbind(seq2, 0)):
            cost = []
            for i in s1:
                row = []
                for j in s2:
                    d = torch.norm((i - j), 2)
                    d = torch.squeeze(d)  # B, 1
                    row.append(d)
                row = torch.squeeze(torch.stack(row))
                cost.append(row)
            cost = torch.stack(cost, 1)
            cost = torch.squeeze(cost)
            cost = cost.numpy()
            row_ind, col_ind = linear_sum_assignment(cost)
            R1.append(torch.stack([s1[i, :] for i in row_ind], 1))
            R2.append(torch.stack([s2[i, :] for i in col_ind], 1))

        reorder_1 = torch.stack(R1, 0)
        reorder_2 = torch.stack(R2, 0)
        return reorder_1, reorder_2


if __name__ == '__main__':
    import numpy as np

    x = [[[0, 1], [0, 0], [1, 0]],
         [[0, 1], [0, 0], [1, 0]]]
    y = [[[0, 1], [0, 0]],
         [[0, 1], [0, 0]]]
    x = np.array(x).astype(np.float64)
    y = np.array(y).astype(np.float64)

    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    hung = Hungarian()
    a, b = hung(x, y)
    print(a)
    print(b)
