import pickle as pickle


#2-7
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
    
data = unpickle("D:\PythonSVM\cifar-10-batches-py\data_batch_1")
data2 = unpickle("D:\PythonSVM\cifar-10-batches-py\data_batch_2")
data3 = unpickle("D:\PythonSVM\cifar-10-batches-py\data_batch_3")
data4 = unpickle("D:\PythonSVM\cifar-10-batches-py\data_batch_4")
data5 = unpickle("D:\PythonSVM\cifar-10-batches-py\data_batch_5")
data_test = unpickle("D:\PythonSVM\cifar-10-batches-py\data_test_batch")
for key in data.keys():
    if key == b'batch_label':
        print('xd')
    else:
        data[key].extend(data2[key])
        data[key].extend(data3[key])
        data[key].extend(data4[key])
        data[key].extend(data5[key])
    

print(data)