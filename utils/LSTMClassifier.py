# import torch.nn as nn
# import torch.nn.functional as F
# import torch



# class LSTMClassifier(nn.Module):

#     def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, batch_size, use_gpu):
#         super(LSTMClassifier, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.batch_size = batch_size
#         self.use_gpu = use_gpu

#         self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim)
#         self.hidden2label = nn.Linear(hidden_dim, label_size)
#         self.hidden = self.init_hidden()

#     def init_hidden(self):
#         if self.use_gpu:
#             h0 = torch.zeros(1, self.batch_size, self.hidden_dim).cuda()
#             c0 = torch.zeros(1, self.batch_size, self.hidden_dim.cuda())
#         else:
#             h0 = torch.zeros(1, self.batch_size, self.hidden_dim)
#             c0 = torch.zeros(1, self.batch_size, self.hidden_dim)
#         return (h0, c0)

#     def forward(self, sentence):
#         embeds = self.word_embeddings(sentence)
#         x = embeds.view(len(sentence), self.batch_size, -1)
#         lstm_out, self.hidden = self.lstm(x, self.hidden)
#         y  = self.hidden2label(lstm_out[-1])
#         return y
import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size,
                 num_layers=1, dropout=0.0, device='cpu'):
        super(LSTMClassifier, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True  # shape: (batch, seq_len, embedding_dim)
        )

        self.fc = nn.Linear(hidden_dim, label_size)

    def forward(self, input_ids):
        # input_ids: (batch_size, seq_len)
        batch_size = input_ids.size(0)

        embeds = self.embedding(input_ids)  # shape: (batch_size, seq_len, embedding_dim)

        # 初始化 h0 和 c0
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device)

        lstm_out, _ = self.lstm(embeds, (h0, c0))  # output: (batch_size, seq_len, hidden_dim)

        # 使用最后一个时间步的输出作为句子表示
        final_hidden = lstm_out[:, -1, :]  # shape: (batch_size, hidden_dim)

        logits = self.fc(final_hidden)  # shape: (batch_size, label_size)
        return logits
