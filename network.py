import torch
import torch.nn as nn
from config import *

class PID_CNN1D(nn.Module):
    def __init__(self, 
                 n_classes: int = 6, 
                 num_of_feature: int = len(FEATURES_LIST), 
                 max_len: int = 1000,
                 kernal_size: list = [5, 3, 3, 3],
                 dropout: float = 0.5,
                 dense_size: int = 512):
        super(PID_CNN1D, self).__init__()
        self.convnet = nn.Sequential(
                                    nn.Conv1d(num_of_feature, 64, kernal_size[0], padding=1), nn.ReLU(), nn.BatchNorm1d(64),
                                    nn.Conv1d(64, 64, kernal_size[1], padding=1), nn.ReLU(), nn.BatchNorm1d(64), 
                                    nn.MaxPool1d(kernal_size[1], stride=kernal_size[1]), nn.Dropout(dropout),
                                    nn.Conv1d(64, 128, kernal_size[1], padding=1), nn.ReLU(), nn.BatchNorm1d(128), 
                                    nn.Conv1d(128, 128, kernal_size[2], padding=1), nn.ReLU(), nn.BatchNorm1d(128), 
                                    nn.MaxPool1d(kernal_size[2], stride=kernal_size[2]), nn.Dropout(dropout), 
                                    nn.Conv1d(128, 128, kernal_size[3], padding=1), nn.ReLU(),
                                    nn.BatchNorm1d(128), nn.AvgPool1d(kernal_size[3], stride=kernal_size[3])
                                    )
        
        # Infer output shape dynamically
        with torch.no_grad():
            dummy_data = torch.ones(1, num_of_feature, max_len)
            out_shape = self.convnet(dummy_data).view(1, -1).shape[-1]
            del dummy_data

        self.fc = nn.Sequential(nn.Linear(out_shape, dense_size),
                                nn.Dropout(dropout),
                                nn.ReLU(),
                                nn.Linear(dense_size, n_classes),
                                nn.Softmax(dim=1)
                                )

    def forward(self, input_layer):
        output = self.convnet(input_layer)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output
    
    def _class_name(self):
        return "PIDCNN1D"