import torch
import torch.nn as nn
import numpy as np

class PID_CNN1D(nn.Module):
    def __init__(self, 
                 n_classes, 
                 num_of_feature, 
                 max_len,
                 batch_size, 
                 kernal_size,
                 dropout,
                 dense_size):
        super(PID_CNN1D, self).__init__()
        self.convnet = nn.Sequential(
                                    nn.Conv1d(num_of_feature, 64, kernal_size[0], padding=1), nn.ReLU(), nn.BatchNorm1d(64),
                                    nn.Conv1d(64, 64, kernal_size[1], padding=1), nn.ReLU(), nn.BatchNorm1d(64), 
                                    nn.MaxPool1d(kernal_size[1], stride=kernal_size[1]), nn.Dropout(dropout),
                                    nn.Conv1d(64, 128, kernal_size[1], padding=1), nn.ReLU(), nn.BatchNorm1d(128), 
                                    nn.Conv1d(128, 128, kernal_size[2], padding=1), nn.ReLU(), nn.BatchNorm1d(128), 
                                    nn.MaxPool1d(kernal_size[2], stride=kernal_size[2]), nn.Dropout(dropout), 
                                    nn.Conv1d(128, 128, kernal_size[3], padding=1), nn.RReLU(),
                                    nn.BatchNorm1d(128), nn.AvgPool1d(kernal_size[3], stride=kernal_size[3])
                                    )
        
        data = torch.ones([batch_size, num_of_feature, max_len])
        out_shape = self.convnet(data).view(batch_size, -1).shape[-1]
        
        self.fc = nn.Sequential(nn.Linear(out_shape, dense_size),
                                nn.Dropout(dropout),
                                nn.ReLU(),
                                nn.Linear(dense_size, n_classes),
                                nn.Softmax(dim=1)
                                )
        
        del data

    def forward(self, input_layer):
        output = self.convnet(input_layer)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output
    
    def _class_name(self):
        return "PIDCNN1D"
    

class PID_CNN1D_large(nn.Module):
    def __init__(self, n_classes, input_size, num_of_feature):
        super(PID_CNN1D_large, self).__init__()
        self.convnet = nn.Sequential(
                                    nn.Conv1d(num_of_feature, 64, 10, padding=1), nn.ReLU(), nn.BatchNorm1d(64),
                                    nn.Conv1d(64, 64, 10, padding=1), nn.ReLU(), nn.BatchNorm1d(64),
                                    nn.Conv1d(64, 64, 10, padding=1), nn.ReLU(), nn.BatchNorm1d(64), 
                                    nn.MaxPool1d(5, stride=5), nn.Dropout(0.5),
                                    nn.Conv1d(64, 128, 10, padding=1), nn.ReLU(), nn.BatchNorm1d(128),
                                    nn.Conv1d(128, 128, 10, padding=1), nn.ReLU(), nn.BatchNorm1d(128), 
                                    nn.MaxPool1d(5, stride=5), nn.Dropout(0.5),
                                    nn.Conv1d(128, 128, 10, padding=1), nn.ReLU(), nn.BatchNorm1d(128), 
                                    nn.Conv1d(128, 128, 10, padding=1), nn.ReLU(), nn.BatchNorm1d(128), 
                                    nn.MaxPool1d(5, stride=5), nn.Dropout(0.5), 
                                    nn.Conv1d(128, 128, 10, padding=1), nn.RReLU(), nn.BatchNorm1d(128), 
                                    nn.Conv1d(128, 128, 10, padding=1), nn.RReLU(), nn.BatchNorm1d(128), 
                                    nn.AvgPool1d(5, stride=5)
                                    )
        
        self.fc = nn.Sequential(nn.Linear(128*7, 1024),
                                nn.Dropout(0.5),
                                nn.ReLU(),
                                nn.Linear(1024, n_classes),
                                nn.Softmax(dim=1)
                                )
        self.input_size = input_size

    def forward(self, input_layer):
        output = self.convnet(input_layer)
        print(output.shape)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output
    
    def _class_name(self):
        return "PIDCNN1D-large"