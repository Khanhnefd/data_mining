import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from recommendation.data_util import add_dense_feature


class RecModel(nn.Module):
    def __init__(
        self,
        hidden_size,
        n_items,
        embedding_dim,
        num_features,
        n_layers=1,
        dropout=0.25,
    ):
        super(RecModel, self).__init__()
        self.hidden_size = hidden_size
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(self.n_items, self.embedding_dim, padding_idx=0)
        self.num_features = num_features
        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        # set bidirectional = True for bidirectional
        # self.gru = nn.GRU(input_size = hidden_size, # number of expected feature of input x
        #                   hidden_size = hidden_size, # number of expected feature of hidden state
        #                   num_layers = n_layers, # number of GRU layers
        #                   dropout=(0 if n_layers == 1 else dropout), # dropout probability apply in encoder network
        #                   bidirectional=True # one or two directions.
        #                  )
        self.emb_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            self.embedding_dim + self.num_features, self.hidden_size, self.n_layers
        )
        self.a_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.a_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_t = nn.Linear(self.hidden_size, 1, bias=False)
        self.ct_dropout = nn.Dropout(0.5)
        self.b = nn.Linear(
            self.embedding_dim + self.num_features, 2 * self.hidden_size, bias=False
        )
        self.sf = nn.Softmax()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, input_seq, input_lengths, hidden=None):
        """
        input_seq: Batch input_sequence. Shape: max_len x batch_size
        input_lengths: Batch input lengths. Shape: batch_size
        """
        # Step 1: Convert sequence indexes to embeddings
        # shape: (max_length , batch_size , hidden_size)
        embedded = self.embedding(input_seq)
        # print("embedded: ", embedded)
        # Step 1.2: add item dense feature
        embedded, new_hidden_size = add_dense_feature(
            input_seq, embedded, self.num_features
        )
        # self.hidden_size=new_hidden_size

        # input_lengths = [il + 3 for il in input_lengths]
        # print("embedded after: ", embedded)
        # print("embedded :", embedded.shape)
        # print("input_lengths :", input_lengths)
        # Pack padded batch of sequences for RNN module. Padding zero when length less than max_length of input_lengths.
        # shape: (total_lengths , hidden_size)
        packed = pack_padded_sequence(embedded, input_lengths)
        # print("packed :", packed.data.shape)
        # print("packed :", packed.batch_sizes)

        # Step 2: Forward packed through GRU
        # outputs is output of final GRU layer
        # hidden is concatenate of all hidden states corresponding with each time step.
        # outputs shape: (max_length , batch_size , hidden_size x num_directions)
        # hidden shape: (n_layers x num_directions , batch_size , hidden_size)
        outputs, hidden = self.gru(packed, hidden)
        # print("outputs :", outputs.data.shape)
        # print("hidden :", hidden.shape)

        # Unpack padding. Revert of pack_padded_sequence
        # outputs shape: (max_length , batch_size , hidden_size x num_directions)
        outputs, length = pad_packed_sequence(outputs)
        # print("outputs :", outputs.shape)
        # print("length :", length)

        # Step 3: Global Encoder & Local Encoder
        # num_directions = 1 -->
        # outputs shape:(max_length , batch_size , hidden_size)
        # hidden shape: (n_layers , batch_size , hidden_size)
        # lấy hidden state tại time step cuối cùng
        ht = hidden[-1]  # (batch_size, hidden_size)
        # print("ht :", ht.shape)
        # reshape outputs
        outputs = outputs.permute(1, 0, 2)  # [batch_size, max_length, hidden_size]
        c_global = ht
        # Flatten outputs thành shape: [batch_size * max_length, hidden_size]
        gru_output_flatten = outputs.contiguous().view(-1, self.hidden_size)
        # Thực hiện một phép chiếu linear projection để tạo các latent variable có shape [batch_size, max_length, hidden_size]
        q1 = self.a_1(gru_output_flatten).view(outputs.size())
        # print("q1:", q1)
        # Thực hiện một phép chiếu linear projection để tạo các latent variable có shape [batch_size, max_length, hidden_size]
        q2 = self.a_2(ht)  # (batch_size, hidden_size)
        # print("q2: ", q2)
        # Ma trận mask đánh dấu vị trí khác 0 trên padding sequence.
        mask = torch.where(
            input_seq.permute(1, 0) > 0,
            torch.tensor([1.0], device=self.device),
            torch.tensor([0.0], device=self.device),
        )  # batch_size x max_len
        # Điều chỉnh shape
        q2_expand = q2.unsqueeze(1).expand_as(
            q1
        )  # shape [batch_size, max_len, hidden_size]
        q2_masked = (
            mask.unsqueeze(2).expand_as(q1) * q2_expand
        )  # batch_size x max_len x hidden_size
        # print("q2_masked :", q2_masked)
        # Tính trọng số alpha đo lường similarity giữa các hidden state
        alpha = self.v_t(torch.sigmoid(q1 + q2_masked).view(-1, self.hidden_size)).view(
            mask.size()
        )  # batch_size x max_len
        alpha_exp = alpha.unsqueeze(2).expand_as(
            outputs
        )  # batch_size x max_len x hidden_size
        # print("alpha_exp :", alpha_exp)
        # Tính linear combinition của các hidden state
        c_local = torch.sum(alpha_exp * outputs, 1)  # (batch_size x hidden_size)

        # Véc tơ combinition tổng hợp
        c_t = torch.cat([c_local, c_global], 1)  # batch_size x (2*hidden_size)
        c_t = self.ct_dropout(c_t)
        # Tính scores

        # Step 4: Decoder
        # embedding cho toàn bộ các item
        item_indices = torch.arange(self.n_items).to(self.device)  # n_items

        item_indices = item_indices.unsqueeze(1)  # n_items x 1
        # print("Step 4")
        # print(item_indices)
        # print(item_indices.shape)

        item_embs = self.embedding(item_indices)  # n_items x 1 x embedding_dim

        item_embs, n = add_dense_feature(
            item_indices, item_embs, self.num_features
        )  # n_items x 1 x (embedding_dim+num_features)
        # print("item_embs:", item_embs)
        # print(item_embs.shape)
        # print(n)

        item_embs = item_embs.squeeze(1)  # n_items x (embedding_dim+num_features)
        # print("item_embs squeeze :",item_embs)
        # reduce dimension by bi-linear projection
        B = self.b(item_embs).permute(1, 0)  # (2*hidden_size) x n_items
        scores = torch.matmul(c_t, B)  # batch_size x n_items
        # scores = self.sf(scores)
        return scores
