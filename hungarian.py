import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment


class Hungarian(nn.Module):
    def __init__(self):
        super().__init__()
        self.cosine = torch.nn.modules.distance.CosineSimilarity()

    def forward(self, seq1, seq2):
        # B, S1, D
        # B, S2, D
        zero, cost = torch.zeros_like(seq1[0, 0, :]), []
        # Calculate similarity matrix
        for i in range(seq1.shape[1]):
            row = []
            for j in range(seq2.shape[1]):
                a, b = seq1[:, i, :], seq2[:, j, :]
                d = self.cosine(a, b)
                row.append(d)
            row = torch.stack(row, 1)
            cost.append(row)
        cost = torch.stack(cost, 1)
        # reorder batch items
        reordered = []
        for batchindex, matrix in enumerate(torch.unbind(cost, 0)):
            row_ind, col_ind = linear_sum_assignment(matrix)
            row_remain = set(range(matrix.shape[0])) - set(row_ind)
            col_remain = set(range(matrix.shape[1])) - set(col_ind)
            row_ind = list(row_ind) + list(sorted(row_remain))
            col_ind = list(col_ind) + list(sorted(col_remain))
            maxl = max(len(row_ind), len(col_ind))
            col_ind += [None] * (maxl - len(col_ind))
            row_ind += [None] * (maxl - len(row_ind))
            s1, s2 = seq1[batchindex, :, :], seq2[batchindex, :, :]

            row = torch.stack([(s1[i, :] if i is not None else zero)
                               for i in row_ind], 0)
            col = torch.stack([(s2[i, :] if i is not None else zero)
                               for i in col_ind], 0)
            cat = torch.cat([row, col], 1)
            reordered.append(cat)
        reordered = torch.stack(reordered, 0)
        return reordered


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
