import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn.inits import reset

EPS = 1e-15


class MLPEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, p1=0.1, p2=0.7):
        super(MLPEncoder, self).__init__()
        self.drop_out1 = nn.Dropout(p1)
        self.linear1 = nn.Linear(input_dim, latent_dim)
        self.drop_out2 = nn.Dropout(p2)
        self.linear2 = nn.Linear(latent_dim, latent_dim)

    def forward(self, x):
        x = self.drop_out1(x)
        x = F.relu(self.linear1(x))
        x = self.drop_out2(x)
        x = self.linear2(x)
        return x


class GCNEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, p1=0.1, p2=0.7):
        super(GCNEncoder, self).__init__()
        self.drop_out1 = nn.Dropout(p1)
        self.conv1 = GCNConv(input_dim, latent_dim, cached=True)
        self.drop_out2 = nn.Dropout(p2)
        self.conv2 = GCNConv(latent_dim, latent_dim, cached=True)

    def forward(self, x, edge_index):
        x = self.drop_out1(x)
        x = F.relu(self.conv1(x, edge_index))
        x = self.drop_out2(x)
        x = self.conv2(x, edge_index)
        return x


class Decoder(nn.Module):
    def forward(self, z, edge_index, sigmoid=True):
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value


class GAE(nn.Module):
    """
    This implementation of a graph auto-encoder is based on code from
    https://github.com/rusty1s/pytorch_geometric/blob/master/examples/autoencoder.py
    """
    def __init__(self, encoder, decoder):
        super(GAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        GAE.reset_parameters(self)

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.decoder)

    def encode(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)

    def bce_loss(self, z, pos_train, sampler):

        # Get negative samples
        neg_train = torch.tensor(sampler.negative_edges(z.size(0)).tolist(), dtype=torch.long).to(pos_train.device)

        pos_loss = -torch.log(self.decoder(z, pos_train, sigmoid=True) + EPS).mean()
        neg_loss = -torch.log(1 - self.decoder(z, neg_train, sigmoid=True) + EPS).mean()

        return pos_loss + neg_loss

    def test(self, z, pos_edge_index, neg_edge_index):

        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred)


class CM(nn.Module):

    # Pass hyperparameters
    def __init__(self, input_dim, embedding_dim, hidden_dim, dropout):
        super(CM, self).__init__()

        # Embedding layers
        self.embedding_a = nn.Embedding(input_dim, embedding_dim, padding_idx=0)
        self.embedding_s = nn.Embedding(input_dim, embedding_dim, padding_idx=0)

        # LSTM layers
        self.lstm_a = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True, num_layers=1)
        self.lstm_s = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True, num_layers=1)

        # Affine layers
        self.linear_1 = nn.Linear(4 * hidden_dim, int(hidden_dim / 2))
        self.linear_2 = nn.Linear(int(hidden_dim / 2), 1)

        # Define dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, a, s):
        e_a = self.dropout(self.embedding_a(a))
        o_a, (h_a, c_a) = self.lstm_a(e_a)

        e_s = self.dropout(self.embedding_s(s))
        o_s, (h_s, c_s) = self.lstm_s(e_s)

        h = torch.cat((o_a[:, -1, :], o_s[:, -1, :]), dim=-1)

        t = self.dropout(F.relu(self.linear_1(h)))

        return F.sigmoid(self.linear_2(t)).squeeze()
