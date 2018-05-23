import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import linear_sum_assignment


class Hungarian(nn.Module):
    def __init__(self, batch_dim_seq=True, force_diag_non_zero=True):
        super().__init__()
        self.cosine = torch.nn.modules.distance.CosineSimilarity()
        self.batch_dim_seq = batch_dim_seq
        self.force_diag_non_zero = force_diag_non_zero

    def forward(self, seq1, seq2):
        if not self.batch_dim_seq:
            seq1 = torch.stack(torch.unbind(seq1, 1), 2)
            seq2 = torch.stack(torch.unbind(seq2, 1), 2)
        # B, D, S1
        # B, D, S2
        zero, cost = torch.zeros_like(seq1[0, :, 0]), []
        # Calculate similarity matrix
        for i in range(seq1.shape[2]):
            row = []
            for j in range(seq2.shape[2]):
                a, b = seq1[:, :, i], seq2[:, :, j]
                d = self.cosine(a, b)  # (Batch, )
                row.append(d)
            row = torch.stack(row, 1)  # (Batch, seq2)
            cost.append(row)
        cost = torch.stack(cost, 1)  # Batch, seq1, seq2
        # reorder batch items
        reordered = []
        for batchindex, matrix in enumerate(torch.unbind(cost, 0)):
            cmat = matrix.detach().numpy()
            if self.force_diag_non_zero:
                np.fill_diagonal(cmat, 0.5)
            row_ind, col_ind = linear_sum_assignment(cmat)
            weights = [matrix[r, c] for r, c in zip(row_ind, col_ind)]

            row_remain = set(range(matrix.shape[0])) - set(row_ind)
            col_remain = set(range(matrix.shape[1])) - set(col_ind)
            row_ind = list(row_ind) + list(sorted(row_remain))
            col_ind = list(col_ind) + list(sorted(col_remain))
            maxl = max(len(row_ind), len(col_ind))

            weights += [1] * (maxl - len(weights))
            col_ind += [None] * (maxl - len(col_ind))
            row_ind += [None] * (maxl - len(row_ind))
            s1, s2 = seq1[batchindex, :, :], seq2[batchindex, :, :]

            row = torch.stack([(w * s1[:, i] if i is not None else zero)
                               for i, w in zip(row_ind, weights)], 1)
            col = torch.stack([w * (s2[:, j] if i is not None else zero)
                               for i, w in zip(col_ind, weights)], 1)
            cat = torch.cat([row, col], 0)
            reordered.append(cat)
        reordered = torch.stack(reordered, 0)
        return reordered


class MyNet(nn.Module):
    def __init__(self, idim, edim):
        super().__init__()
        self.conv1 = nn.Conv1d(idim, edim, 3, padding=1)
        self.hung = Hungarian()

    def forward(self, seq):
        x = self.conv1(seq)
        y = self.hung(x, x)
        return y


if __name__ == '__main__':
    # Batch, Dim, Seqlen
    x = np.random.random((16, 300, 200))
    y = np.random.random((16, 300, 100))

    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()

    hung = Hungarian()
    a = hung(x, y)
    print(a.shape)

    print('-'*100)

    net = MyNet(300, 64)
    a = net(y)
    print(a.shape)
