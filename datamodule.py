from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

class DealDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
        self.len = x_data.shape[0]
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

def data_loader(data_X, data_y, batch_size, shuffle=True):
    data = DealDataset(data_X, data_y)
    loader = DataLoader(dataset=data,           
                    batch_size=batch_size, 
                    shuffle=shuffle,
                    num_workers=4, 
                    pin_memory=True)
    return loader

class DataModule(pl.LightningDataModule):
    def __init__(self, data=None, verbose=True, batch_size=None):
        super().__init__()
        
        self.data = data
        self.verbose = verbose
        self.batch_size = batch_size

    def train_dataloader(self):
        return data_loader(self.data['train_x'], self.data['train_y'], self.batch_size)

    def test_dataloader(self):
        return data_loader(self.data['test_x'], self.data['test_y'], self.batch_size, False)

    def val_dataloader(self):
        return data_loader(self.data['valid_x'], self.data['valid_y'], self.batch_size, False)

    def __repr__(self):
        return (f'Dataset(root="{self.root}", '
                f'samples={len(self.samples)}, '
                f'avglen={self.avglen})')
