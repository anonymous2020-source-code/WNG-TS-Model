import numpy as np
from scipy.io import loadmat
import method
from sklearn.model_selection import StratifiedKFold
import SGCN as SGCN_model
import GNN as GNN_model
import CNN1D as CNN1D_model
import MLP as MLP_model
import time


def load_data(dataset):
    path = "../data/"
    data_name_t = dataset + 'T'
    # data_name_f = dataset + 'F'

    path_data_t = path + data_name_t + '.mat'
    # path_data_f = path + data_name_f + '.mat'

    data_t = loadmat(path_data_t)[data_name_t]
    # data_f = loadmat(path_data_f)[data_name_f]
    time11 = time.time()
    data_f = abs(np.fft.fft(data_t[:, :])) / 256
    time12 = time.time()
    fft_time = (time12 - time11)
    print("FFT time : %3.8f" % (fft_time))
    return data_t, data_f, fft_time


def convert_to_graph(data_t, data_f, func):
    samples = data_f.shape[0]

    time1 = time.time()
    adj_f = func(data_f)
    time2 = time.time()

    time3 = time.time()
    adj_t = func(data_t)
    time4 = time.time()

    adj_f_time = (time2 - time1) / samples
    adj_t_time = (time4 - time3) / samples

    print("adj_F convert time : %3.8f, adj_T convert time: %3.8f" % (adj_f_time, adj_t_time))
    print("Data load Finished!")
    label1 = np.zeros(int(samples / 2))
    label2 = np.ones(int(samples / 2))
    label = np.concatenate((label1, label2), axis=0)
    return adj_t, adj_f, label, adj_t_time, adj_f_time


def run_main(dataset):
    print("DataSet:{}".format(dataset))
    # iteration flop
    for iteration in range(1):
        data_t, data_f, fft_time = load_data(dataset)
        samples = data_f.shape[0]

        # methods = {'WNG': method.overlook_WNG, 'WOG': method.overlookg, 'WRG': method.overlook_WRG, 'V': method.LPvisibility_v, 'LV': method.LPvisibility_lv, 'H': method.LPhorizontal_h, 'LH': method.LPhorizontal_lh}
        methods = {'WNG': method.overlook_WNG}
        # method flop
        for method_name in methods:
            print("Method:{}".format(method_name))
            adj_t, adj_f, label, adj_t_time, adj_f_time = convert_to_graph(data_t, data_f, methods[method_name])

            with open('save_result/{}_{}_convert_time_TSCNN1D.txt'.format(method_name, dataset), 'a+') as f:
                f.write(str(iteration) + '\t' + str(format(adj_t_time, '.5f')) + '\t' + str(
                    format(adj_f_time, '.5f')) + '\t' + str(format(fft_time, '.5f'))
                        + '\n')

            skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
            i = 0

            # fold flop
            for train_index, val_index in skf.split(data_f, label):
                i += 1
                print("Fold:{}".format(i))
                train_f, train_t, train_label = adj_f[train_index], adj_t[train_index], label[train_index]
                val_f, val_t, val_label = adj_f[val_index], adj_t[val_index], label[val_index]

                train_time, val_time, val_spe, val_rec, val_acc = CNN1D_model.model_main(train_t, train_f, train_label,
                                                                                        val_t, val_f, val_label, i,
                                                                                        method_name, dataset)

                train_time_mean = train_time / samples
                val_time_mean = val_time / samples

                print("Train time:%3.8f" % train_time_mean)
                print("Val time:%3.8f" % val_time_mean)
                print("Val accuracy:%.3f" % val_acc)
                print("Val precision:%.3f" % val_spe)
                print("val recall:%.3f" % val_rec)

                with open('save_result/{}_{}_train_time_TSCNN1D.txt'.format(method_name, dataset), 'a+') as f:
                    f.write(str(iteration) + '\t' + str(i) + '\t' + str(format(train_time_mean, '.5f')) + '\t' +
                            str(format(val_time_mean, '.5f')) + '\t' + str(format(val_spe, '.5f')) +
                            '\t' + str(format(val_rec, '.5f')) + '\t' + str(format(val_acc, '.5f')) + '\n')


if __name__ == '__main__':
    run_main('AE')