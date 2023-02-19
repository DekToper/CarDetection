def data_separation(data):
    train_size = int(len(data) * .7)
    test_size = int(len(data) * .1)
    val_size = int(len(data) * .2)

    train = data.take(train_size)
    test = data.skip(train_size).take(test_size)
    val = data.skip(train_size + test_size).take(val_size)

    return train, test, val
