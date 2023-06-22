import argparse
import sys
import torch
import time
import scipy.io as sio
import numpy as np
from torch.utils.data import TensorDataset, DataLoader



class CNNnet(torch.nn.Module):
      def __init__(self, node_number, batch_size, k_hop):
          super(CNNnet,self).__init__()
          self.node_number = node_number
          self.batch_size = batch_size
          self.k_hop = k_hop
          self.aggregate_weightT = torch.nn.Parameter(torch.ones(1, 1, node_number))
          self.aggregate_weightF = torch.nn.Parameter(torch.ones(1, 1, node_number))
          self.conv1 = torch.nn.Sequential(
              torch.nn.Conv1d(in_channels=1,
                              out_channels=8,
                              kernel_size=3,
                              stride=1,
                              padding=1),
              torch.nn.BatchNorm1d(8),
              torch.nn.ReLU(),
              torch.nn.MaxPool1d(kernel_size=2),
              torch.nn.Dropout(0.2),
          )
          self.conv2 = torch.nn.Sequential(
              torch.nn.Conv1d(8,16,3,1,1),
              torch.nn.BatchNorm1d(16),
              torch.nn.ReLU(),
              torch.nn.MaxPool1d(kernel_size=2),
              torch.nn.Dropout(0.2),
          )
          self.conv3 = torch.nn.Sequential(
              torch.nn.Conv1d(16,32,9,1,4),
              torch.nn.BatchNorm1d(32),
              torch.nn.ReLU(),
              torch.nn.MaxPool1d(kernel_size=2),
              torch.nn.Dropout(0.2),
          )
          self.conv4 = torch.nn.Sequential(
              torch.nn.Conv1d(32,32,9,1,4),
              torch.nn.BatchNorm1d(32),
              torch.nn.ReLU(),
              torch.nn.MaxPool1d(kernel_size=2),
              torch.nn.Dropout(0.2),
          )
          self.mlp1 = torch.nn.Sequential(
              torch.nn.Linear(16*32,64),#12
              torch.nn.Dropout(0.5),
          )

          self.conv11 = torch.nn.Sequential(
              torch.nn.Conv1d(in_channels=1,
                              out_channels=8,
                              kernel_size=3,
                              stride=1,
                              padding=1),
              torch.nn.BatchNorm1d(8),
              torch.nn.ReLU(),
              torch.nn.MaxPool1d(kernel_size=2),
              torch.nn.Dropout(0.2),
          )
          self.conv21 = torch.nn.Sequential(
              torch.nn.Conv1d(8,16,3,1,1),
              torch.nn.BatchNorm1d(16),
              torch.nn.ReLU(),
              torch.nn.MaxPool1d(kernel_size=2),
              torch.nn.Dropout(0.2),
          )
          self.conv31 = torch.nn.Sequential(
              torch.nn.Conv1d(16,32,9,1,4),
              torch.nn.BatchNorm1d(32),
              torch.nn.ReLU(),
              torch.nn.MaxPool1d(kernel_size=2),
              torch.nn.Dropout(0.2),
          )
          self.conv41 = torch.nn.Sequential(
              torch.nn.Conv1d(32,32,9,1,4),
              torch.nn.BatchNorm1d(32),
              torch.nn.ReLU(),
              torch.nn.MaxPool1d(kernel_size=2),
              torch.nn.Dropout(0.2),
          )
          self.mlp11 = torch.nn.Sequential(
              torch.nn.Linear(16*32,64),#12
              torch.nn.Dropout(0.5),
          )

          self.mlp2 = torch.nn.Linear(128,2)
      def forward(self, x_T, x_F):
      
          tmp_x_T = x_T
          for _ in range(self.k_hop):
              tmp_x_T = torch.matmul(tmp_x_T, x_T)
          x_T = torch.matmul(self.aggregate_weightT, tmp_x_T)
          x_T = self.conv1(x_T)
          x_T = self.conv2(x_T)
          x_T = self.conv3(x_T)
          x_T = self.conv4(x_T)
          x_T = self.mlp1(x_T.view(x_T.size(0),-1))
          
          tmp_x_F = x_F
          for _ in range(self.k_hop):
              tmp_x_F = torch.matmul(tmp_x_F, x_F)
          x_F = torch.matmul(self.aggregate_weightF, tmp_x_F)
          x_F = self.conv11(x_F)
          x_F = self.conv21(x_F)
          x_F = self.conv31(x_F)
          x_F = self.conv41(x_F)
          x_F = self.mlp11(x_F.view(x_F.size(0),-1))
          #print(x_F.shape)
          x = torch.cat((x_T, x_F), 1)
          #print(x.shape)
          x = self.mlp2(x)
          return x


def model_main(train_T, train_F, train_label, val_T, val_F, val_label, seed, method_name, dataset):

    parser = argparse.ArgumentParser(description='PyTorch graph convolutional neural net for whole-graph classification')
    #parser.add_argument('--dataset', type=str, default="TF_SGCN/DET_WOGTF.mat", help='path of the dataset (default: data/data.mat)')
    parser.add_argument('--node_number', type=int, default=256, help='node number of graph (default: 256)')#200
    parser.add_argument('--batch_size', type=int, default=32, help='number of input size (default: 128)')
    parser.add_argument('--k_hop', type=int, default=2, help='times of aggregate (default: 1)')

    args = parser.parse_args()

    train_T = np.array(train_T, dtype=float)
    train_F = np.array(train_F, dtype=float)
    val_T = np.array(val_T, dtype=float)
    val_F = np.array(val_F, dtype=float)
    train_label = np.array(train_label, dtype=int)
    val_label = np.array(val_label, dtype=int)


    train_T = torch.FloatTensor(train_T)
    train_F = torch.FloatTensor(train_F)
    val_T = torch.FloatTensor(val_T)
    val_F = torch.FloatTensor(val_F)
    train_label = torch.LongTensor(train_label)
    val_label = torch.LongTensor(val_label)

    train_set = TensorDataset(train_T, train_F, train_label)
    val_set = TensorDataset(val_T, val_F, val_label)

    # batch_size = 128
    torch.manual_seed(seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = CNNnet(args.node_number, args.batch_size, args.k_hop)
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)   # optimize all cnn parameters
    
    best_acc = 0.0
    best_val_spe = 0.0
    best_val_rec = 0.0

    num_epoch = 100
    train_epoch_time = []
    val_epoch_time = []
    for epoch in range(num_epoch):
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0

        model.train()
        train_start_time = time.time()
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            t1 = time.time()
            train_pred = model(data[0],data[1]) 
            # print(train_pred.size())
            # print(data[1].size())

            batch_loss = loss(train_pred, data[2])  
            batch_loss.backward()  
            optimizer.step()  

            train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[2].numpy())
            train_loss += batch_loss.item()
            t2 = time.time()

            # progress = ('#' * int(float(i) / len(train_loader) * 40)).ljust(40)
            # print('[%03d/%03d] %2.2f sec(s) | %s |' % (epoch + 1, num_epoch, \
            #                                       (time.time() - epoch_start_time), progress), end='\r', flush=True)
        train_end_time = time.time()
        train_epoch_time.append(train_end_time - train_start_time)

        model.eval()

        val_TP = 1.0
        val_TN = 1.0
        val_FN = 1.0
        val_FP = 1.0
        predict_total = []
        label_total = []
        val_start_time = time.time()
        for i, data in enumerate(val_loader):
            val_pred = model(data[0],data[1])
            t3 = time.time()
            batch_loss = loss(val_pred, data[2])
    
            predict_val = np.argmax(val_pred.cpu().data.numpy(), axis=1)
            predict_total = np.append(predict_total, predict_val)
            label_val = data[2].numpy()
            label_total = np.append(label_total, label_val)
            
            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[2].numpy())
            val_loss += batch_loss.item()
            t4 = time.time()
            
            
        val_end_time = time.time()
        val_epoch_time.append(val_end_time - val_start_time)

        val_TP = ((predict_total == 1) & (label_total == 1)).sum().item()
        val_TN = ((predict_total == 0) & (label_total == 0)).sum().item()
        val_FN = ((predict_total == 0) & (label_total == 1)).sum().item()
        val_FP = ((predict_total == 1) & (label_total == 0)).sum().item()
    
        # p = TP/(TP + FP + 0.01)
        train_time = t2 - t1
        test_time = t4 - t3
        val_spe = val_TN/(val_FP + val_TN + 0.0001)
        val_rec = val_TP/(val_TP + val_FN + 0.0001)
        test_acc = (val_TP+val_TN)/(val_FP + val_TN + val_TP + val_FN + 0.0001)
            
        val_acc = val_acc / val_set.__len__()

        print('%3.5f %3.5f %3.8f %3.5f %3.5f %3.8f' % (train_acc / train_set.__len__(), train_loss, train_time/32,
                                                       val_acc, val_loss, test_time/32))
        with open('save_time_TF/{}_{}_epoch_TSSGCN.txt'.format(method_name, dataset), 'a+') as f:
              f.write(str(epoch) + '\t' + str(format(train_acc / train_set.__len__(),'.5f')) + '\t' +
              str(format(train_loss,'.5f')) + '\t' + str(format(train_time/32,'.8f')) +
              '\t' + str(format(val_acc,'.5f')) + '\t' + str(format(val_loss,'.5f')) + '\t' +
              str(format(test_time/32,'.8f')) + '\n')

        if (val_acc > best_acc):
            torch.save(model.state_dict(), 'save/{}_{}_TSSGCN.pth'.format(method_name, dataset))
            best_acc = val_acc
            best_val_rec = val_rec
            best_val_spe = val_spe
            # print('Model Saved!')

    #for name, param in model.named_parameters():
    #    if param.requires_grad:
    #        print(param[0])

    return np.mean(train_epoch_time), np.mean(val_epoch_time), best_val_spe, best_val_rec, best_acc,

