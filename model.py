import torch
from torch import nn
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target, index

    def __len__(self):
        return len(self.dataset)


class Alex_NeuralNet_Meta(nn.Module):
    def __init__(self, hidden_size, lin_size, out_size, features, params, embedding_matrix):
        super(Alex_NeuralNet_Meta, self).__init__()

        # Initialize some parameters for your model
        self.hidden_size = hidden_size
        drp = 0.1

        # Layer 1: Word2Vec Embeddings.
        self.embedding = nn.Embedding(params.max_features, params.embed_size)
        self.embedding.weight = nn.Parameter(
            torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        # Layer 2: Dropout1D(0.1)
        self.embedding_dropout = nn.Dropout2d(0.1)

        # Layer 3: Bidirectional CuDNNLSTM
        self.lstm = nn.LSTM(params.embed_size, hidden_size,
                            bidirectional=True, batch_first=True)

        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

        # Layer 4: Bidirectional CuDNNGRU
        self.gru = nn.GRU(hidden_size*2, hidden_size,
                          bidirectional=True, batch_first=True)

        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

        # Layer 7: A dense layer
        self.linear = nn.Linear(hidden_size*6 + features.shape[1], lin_size)
        self.relu = nn.ReLU()

        # Layer 8: A dropout layer
        self.dropout = nn.Dropout(drp)

        # Layer 9: Last linear layer
        self.linear2 = nn.Linear(lin_size, out_size)

        # Layer 10: Softmax layer for outputting the class probabilities
        self.out = nn.Softmax(dim=1)

    def forward(self, x):
        '''
        here x[0] represents the first element of the input that is going to be passed. 
        We are going to pass a tuple where first one contains the sequences(x[0])
        and the second one is a additional feature vector(x[1])
        '''
        h_embedding = self.embedding(x[0])
        # Based on comment by Ivank to integrate spatial dropout.
        embeddings = h_embedding.unsqueeze(2)    # (N, T, 1, K)
        embeddings = embeddings.permute(0, 3, 2, 1)  # (N, K, 1, T)
        # (N, K, 1, T), some features are masked
        embeddings = self.embedding_dropout(embeddings)
        embeddings = embeddings.permute(0, 3, 2, 1)  # (N, T, 1, K)
        h_embedding = embeddings.squeeze(2)  # (N, T, K)
        #h_embedding = torch.squeeze(self.embedding_dropout(torch.unsqueeze(h_embedding, 0)))

        #print("emb", h_embedding.size())
        h_lstm, _ = self.lstm(h_embedding)
        # print("lst",h_lstm.size())
        h_gru, hh_gru = self.gru(h_lstm)
        hh_gru = hh_gru.view(-1, 2*self.hidden_size)
        #print("gru", h_gru.size())
        #print("h_gru", hh_gru.size())

        # Layer 5: is defined dynamically as an operation on tensors.
        avg_pool = torch.mean(h_gru, 1)
        max_pool, _ = torch.max(h_gru, 1)
        #print("avg_pool", avg_pool.size())
        #print("max_pool", max_pool.size())

        # the extra features you want to give to the model
        f = torch.tensor(x[1], dtype=torch.float).cuda()
        #print("f", f.size())

        # Layer 6: A concatenation of the last state, maximum pool, average pool and
        # additional features
        conc = torch.cat((hh_gru, avg_pool, max_pool, f), 1)
        #print("conc", conc.size())

        # passing conc through linear and relu ops
        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        conc = self.linear2(conc)
        out = self.out(conc)
        # return the final output
        return out
