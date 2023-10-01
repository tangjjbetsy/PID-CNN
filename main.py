from network import *
from torch import optim
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from sklearn.metrics import classification_report

import os
import wandb
import argparse
import torchmetrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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


class MyLightningModule(pl.LightningModule):
    def __init__(self, 
                 net, 
                 config,
                 weights = None):
        
        super().__init__()
        
        self.net = net
        self.loss_fn = nn.CrossEntropyLoss(weight = torch.FloatTensor(weights))
        self.train_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.test_f1_score = torchmetrics.F1Score()
        self.learning_rate = config.learning_rate
        self.weight_decay = config.weight_decay
        # self.save_hyperparameters()

    
    def forward(self, x):
        return self.net(x)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 2000, eta_min=5e-5)
        return optimizer
    
    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.lr_scheduler.step()  
    
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        inputs = torch.transpose(inputs, 1, 2).float()
        outputs = self(inputs)
        loss = self.loss_fn(outputs, labels)
        
        acc = self.train_accuracy(outputs.argmax(dim=1), labels)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, on_step=False)
        self.log('train_acc', acc, on_epoch=True, prog_bar=True, on_step=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        inputs = torch.transpose(inputs, 1, 2).float()
        outputs = self(inputs)
        loss = self.loss_fn(outputs, labels)
        
        acc = self.val_accuracy(outputs.argmax(dim=1), labels)
        self.log('val_loss', loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, on_step=False, prog_bar=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        inputs = torch.transpose(inputs, 1, 2).float()
        outputs = self(inputs)
        loss = self.loss_fn(outputs, labels)
        
        acc = self.test_accuracy(outputs.argmax(dim=1), labels)
        f1_score = self.test_f1_score(outputs.argmax(dim=1), labels)
        
        self.log('test_loss', loss, on_epoch=True, prog_bar=True, on_step=False)
        self.log('test_acc', acc, on_epoch=True, prog_bar=True, on_step=False)
        self.log('test_f1', f1_score, on_epoch=True, prog_bar=True, on_step=False)

        return loss    

def get_args():
    parser = argparse.ArgumentParser(description='')
    
    parser.add_argument("--model", type=str, default="PID_CNN1D")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--num_of_features", type=int, default=13)
    parser.add_argument("--num_of_performers", type=int, default=6)
    parser.add_argument("--cuda_devices", nargs='+', default=["0"], help="CUDA device ids")
    parser.add_argument("--save_path", type=str, default="test_results")
    parser.add_argument("--ckpt_path", type=str, default=None)

    parser.add_argument("--sweep_name", type=str, default=None)
    parser.add_argument("--sweep_count", type=int, default=5)
    
    args = parser.parse_args()
    return args

def train(config):
    wandb.init(config=config)
    config = wandb.config
    data = np.load(config.data_path)
    max_len = data["train_x"].shape[1]
    
    net = eval(config.model)(config.num_of_performers, 
                                int(config.data_path.split("/")[-1].split("_")[1]),
                                max_len,
                                config.batch_size,
                                config.kernal_size,
                                config.dropout,
                                config.dense_size)
                                
    logger = WandbLogger(log_model=True)
    
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", save_last=True) 
    lr_monitor = LearningRateMonitor(logging_interval="step")
    # el_stopping = EarlyStopping(monitor="val_loss", patience=50)
    
    weights = np.unique(data['train_y'],return_counts=True)[1] / data['train_y'].shape[0]
    model = MyLightningModule(net, config, weights)
    
    logger.watch(model.net)
    
    if config.ckpt_path != None:
        model = model.load_from_checkpoint(config.ckpt_path)
    
    datamodule = DataModule(data, batch_size=config.batch_size)
    
    trainer = pl.Trainer(max_epochs=config.epochs, 
                        logger=logger,
                        accelerator='gpu', 
                        devices=1,
                        precision=16,
                        enable_progress_bar=True,
                        log_every_n_steps=10,
                        callbacks=[checkpoint_callback, lr_monitor])
    
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule, ckpt_path=checkpoint_callback.best_model_path)
    

def test(config):
    wandb.init(config=config)
    config = wandb.config
    data = np.load(config.data_path)
    max_len = data["train_x"].shape[1]
    
    net = eval(config.model)(config.num_of_performers, 
                                config.num_of_features,
                                max_len,
                                config.batch_size,
                                config.kernal_size,
                                config.dropout,
                                config.dense_size)
    weights = np.unique(data['train_y'],return_counts=True)[1] / data['train_y'].shape[0]
    
    model = MyLightningModule(net, config, weights)
    data = data_loader(data['test_x'], data['test_y'], batch_size=config.batch_size)
    
    model = model.load_from_checkpoint(config.ckpt_path)
    preds = []
    labels = []
    
    model.eval()
    for batch, label in data:
        input = torch.transpose(torch.tensor(batch), 1, 2).float()
        with torch.no_grad():
            outputs = model(input)
            outputs = torch.argmax(outputs, dim=-1)
    
            for i in outputs:
                preds += outputs.tolist()
                labels += label.tolist()
    
    PERFROMER = [
            "Alfred Brendel",
            "Claudio Arrau",
            "Daniel Barenboim",
            "Friedrich Gulda",
            "Sviatoslav Richter",
            "Wilhelm Kempff"
    ]
    
    # Save classification report
    sns.set_theme(style="darkgrid")

    text = classification_report(labels, preds, labels=np.arange(config.num_of_performers), output_dict=True, zero_division=0)
    plt.title('Classification Report for the Performer Identification')
    sns.heatmap(pd.DataFrame(text).iloc[:-1, :].T, annot=True, cmap='Greens')
    plt.savefig(os.path.join(config.save_path, "classification_report.png"), bbox_inches='tight')
    
    # Save confustion matrix
    df = pd.DataFrame(np.stack([preds, labels], axis=1), columns=['pred', 'real'])
    df_confusion = pd.crosstab(df['real'], df['pred'], rownames=['Actual'], colnames=['Predicted'],dropna=False,  margins=True)
    plt.clf()
    plt.figure(figsize=(7,5))
    df_confusion = df_confusion.iloc[0:-1,0:-1]/df_confusion.iloc[-1]
    df_confusion = df_confusion.apply(lambda x: round(x, 2))
    df_confusion = df_confusion.iloc[:, 0:-1]
    sns.heatmap(df_confusion, cmap="Blues", annot=True, xticklabels=PERFROMER)
    plt.title('Confusion Matrix for the Performer Identification')
    plt.savefig(os.path.join(config.save_path, "confusion_matrix.png"), bbox_inches='tight')
                
if __name__ == "__main__":
    args = get_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(args.cuda_devices)
    config = {
            'epochs': 1500,
            'batch_size': 16,
            'learning_rate': 8e-5,
            'weight_decay': 1e-7,
            "data_path": "/import/c4dm-04/jt004/ATEPP-data-exp/processed_data/id_13_full_large_0.npz",
            "ckpt_path": args.ckpt_path,
            "save_path": args.save_path,
            "model": "PID_CNN1D",
            "num_of_performers": args.num_of_performers,
            "num_of_features": args.num_of_features,
            "kernal_size": [5, 5, 5, 3],
            "dropout": 0.5,
            "dense_size": 512
        }
    
    if args.mode == "train":
        train(config)
    elif args.mode == "test":
        test(config)
        
        
        