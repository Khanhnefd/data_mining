import torch
from torch.nn.utils.rnn import pad_sequence
from recommendation.util import get_tracks_feature_data

feature_data = get_tracks_feature_data()


def collate_fn(data):
    """
    Hàm số này sẽ được sử dụng để pad session về max length
    Args:
      data: batch truyền vào
    return:
      batch data đã được pad length có shape maxlen x batch_size
    """
    # Sort batch theo độ dài của input_sequence từ cao xuống thấp
    data.sort(key=lambda x: len(x[0]), reverse=True)
    lens = [len(sess) for sess, label in data]
    labels = []
    # Padding batch size
    # padded_sesss = torch.zeros(len(data), max(lens)).long()
    # for i, (sess, label) in enumerate(data):
    #     padded_sesss[i,:lens[i]] = torch.LongTensor(sess)
    #     labels.append(label)

    # print(padded_sesss)
    inputs = []
    for s, l in data:
        inputs.append(torch.LongTensor(s))
        labels.append(l)
    padded_sesss = pad_sequence(inputs)

    # Transpose dữ liệu từ batch_size x maxlen --> maxlen x batch_size
    # padded_sesss = padded_sesss.transpose(0,1)
    return padded_sesss, torch.tensor(labels).long(), lens


def add_dense_feature(input_seq, embeddings, num_features):
    """
    input_seq: (maxlen x batch_size)
    embeddings:  embeddings to add dense feature ==> (max_length , batch_size , hidden_size)
    """

    input_expand = input_seq.view(-1, 1).expand(-1, num_features).clone().float()
    # print("input_expand = ", input_expand)
    for i, input_item in enumerate(input_seq.flatten()):
        if input_item.item() == 0:
            # print("input_item = ", input_item)
            # print(type(input_item.item()))
            continue
        input_expand[i] = torch.tensor(
            feature_data.get(input_item.item(), [0] * num_features)
        )
        # print("get feature:", input_expand[i])

    input_expand = input_expand.view(
        input_seq.shape[0], input_seq.shape[1], num_features
    )
    outputs = torch.cat((input_expand, embeddings), 2)
    new_hidden_size = embeddings.shape[2] + num_features

    return outputs, new_hidden_size
