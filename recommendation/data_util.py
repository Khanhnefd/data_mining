import torch
from torch.nn.utils.rnn import pad_sequence
from recommendation.util import get_tracks_feature_data, get_track_ids
import numpy as np
import os
import pickle
from app.function import (
    connect_mongo,
    insert_web_data,
    get_data_date,
    insert_listen_history,
    get_user_streaming_history,
)
import pandas as pd
from datetime import datetime

feature_data = get_tracks_feature_data()
db_user_history = "mev_user_history"
list_track_id = get_track_ids()


def load_data(
    root="data",
    valid_portion=0.1,
    maxlen=20,
    sort_by_len=False,
    train_set=None,
    test_set=None,
):
    """Load dataset từ root
    root: folder dữ liệu train, trong trường hợp train_set, test_set tồn tại thì không sử dụng train_set và test_set
    valid_portion: tỷ lệ phân chia dữ liệu validation/train
    maxlen: độ dài lớn nhất của sequence
    sort_by_len: có sort theo chiều dài các session trước khi chia hay không?
    train_set: training dataset
    test_set:  test dataset
    """

    # Load the dataset
    if train_set is None and test_set is None:
        path_train_data = os.path.join(root, "train.pkl")
        path_test_data = os.path.join(root, "test.pkl")
        with open(path_train_data, "rb") as f1:
            train_set = pickle.load(f1)

        with open(path_test_data, "rb") as f2:
            test_set = pickle.load(f2)

    if maxlen:
        new_train_set_x = []
        new_train_set_y = []
        # Lọc dữ liệu sequence đến maxlen
        for x, y in zip(train_set[0], train_set[1]):  # x = [214652220], y = 214840483
            if len(x) < maxlen:
                new_train_set_x.append(x)
                new_train_set_y.append(y)
            else:
                new_train_set_x.append(x[:maxlen])
                new_train_set_y.append(y)
        train_set = (new_train_set_x, new_train_set_y)
        del new_train_set_x, new_train_set_y

        new_test_set_x = []
        new_test_set_y = []
        for xx, yy in zip(test_set[0], test_set[1]):
            if len(xx) < maxlen:
                new_test_set_x.append(xx)
                new_test_set_y.append(yy)
            else:
                new_test_set_x.append(xx[:maxlen])
                new_test_set_y.append(yy)
        test_set = (new_test_set_x, new_test_set_y)
        del new_test_set_x, new_test_set_y

    # phân chia tập train thành train và validation
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype="int32")
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1.0 - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    (test_set_x, test_set_y) = test_set

    # Trả về indices thứ tự độ dài của mỗi phần tử trong seq
    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    """
        len_argsort([[1, 2, 3, 4], [1], [1, 2, 3, 4, 5, 6]])
        ==> [1, 0, 2]
    """

    # Sắp xếp session theo độ dài tăng dần
    if sort_by_len:
        sorted_index = len_argsort(test_set_x)
        test_set_x = [test_set_x[i] for i in sorted_index]
        test_set_y = [test_set_y[i] for i in sorted_index]

        sorted_index = len_argsort(valid_set_x)
        valid_set_x = [valid_set_x[i] for i in sorted_index]
        valid_set_y = [valid_set_y[i] for i in sorted_index]

    train = (train_set_x, train_set_y)
    valid = (valid_set_x, valid_set_y)
    test = (test_set_x, test_set_y)
    return train, valid, test


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


def _preprocess_sess_dict(sessDict):
    sessDict = list(sessDict.items())

    # print(sessDict)

    itemIds = [item[1] for item in sessDict]
    inp_seq = []
    labels = []

    for i in range(len(sessDict)):
        if i >= 1:
            inp_seq += [itemIds[:i]]
            labels += [itemIds[i]]
    # print('-------------process----------')
    # print(inp_seq)
    # print(labels)

    return inp_seq, labels, itemIds


def _preprocess_data(data_sess):
    inp_seqs = []
    labels = []
    sequences = []
    sessIds = list(data_sess.index)
    # print(len(sessIds))
    for sessId in sessIds:
        sessDict = data_sess.loc[sessId, 'Sequence']
        # print("sessDict", sessDict)
        inp_seq, label, sequence = _preprocess_sess_dict(sessDict)
        inp_seqs += inp_seq
        labels += label
        sequences += sequence
    # print(len(inp_seqs))
    return inp_seqs, labels, sequences


def get_data_continue_train(seq_len: int = 20):
    mongo_client = connect_mongo()
    mev_history = mongo_client.get_database(db_user_history)

    collections = mev_history.list_collection_names()

    # collections = ['01893ea5-8bb1-77ee-aa46-5ba6ff236c26', '018bf162-e301-7e40-ae4f-dbcaece67ff2']
    data_total = []
    for coll in collections:
        # print(f"user id: {coll}")
        collection = mev_history[coll]
        data_raw = list(
            collection.find(
                {"isTrain": {"$in": [False, None]}, "trackId": {"$in": list_track_id}},
                {"trackId": 1, "timestamp": 1, "_id": 1},
            )
        )
        if len(data_raw) == 0:
            print("No data")
            continue
        else:
            for d in data_raw:
                d.update({"timestamp": d['timestamp'].strftime(r"%Y/%m/%d-%H:%M:%S")})
                d.update({"trackId": list_track_id.index(d["trackId"])})
            data_total.extend(data_raw)

            id_list  = [d['_id'] for d in data_raw]

            collection.update_many(
                {
                    "_id": {"$in": id_list}
                },
                {
                     "$set": {"isTrain": True} 
                }
            )

    print("total data len: ", len(data_total))
    data = []
    for i in range(0, len(data_total), seq_len):
        data_seq = data_total[i : i + seq_len]
        # print("data seq:", data_seq)
        data_seq = {d['timestamp']: d['trackId'] for d in data_seq}
        # print("data seq:", data_seq)
        data.append([data_seq])

    # print("data: ",data)
    # print("-----------------")
    data_df = pd.DataFrame(data)
    data_df.columns = ['Sequence']
    # print("data_df: ",data_df)
    # data_df['Sequence'] = dict(data_df['trackId']: data_df['timestamp'])

    # data_df['Sequence'] = [dict({key: val}) for key, val in zip(data_df.trackId, data_df.timestamp)]

    # data_df = data_df.drop(['trackId', 'timestamp'], axis=1)
    # print(data_df)
    # data_df.set_index('userId', inplace=True)

    train_data = data_df.sample(frac=0.9)
    test_data = data_df.drop(train_data.index)

    train_inp_seqs, train_labs, train_sequences = _preprocess_data(
        train_data
    )
    test_inp_seqs, test_labs, test_sequences = _preprocess_data(
        test_data
    )

    train = (train_inp_seqs, train_labs)
    test = (test_inp_seqs, test_labs)

    # print("---------result--------")
    # print(len(train[0]))
    # print(len(test[0]))
    # print("---------sample--------")
    # print(train[0][3])
    # print(train[1][3])

    return train, test



if __name__ == "__main__":
    get_data_continue_train()
