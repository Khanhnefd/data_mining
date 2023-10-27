import torch
from torch.nn.utils.rnn import pad_sequence

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