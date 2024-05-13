from network import PID_CNN1D
from datamodule import DataModule, data_loader
from config import FEATURES_LIST

from torch import optim
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from sklearn.metrics import classification_report

import os
import torch
import wandb
import argparse
import torchmetrics

import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import pytorch_lightning as pl

class PIDLightningModule(pl.LightningModule):
    def __init__(self, 
                 net, 
                 config,
                 weights = None):
        
        super().__init__()
        
        self.net = net
        self.loss_fn = nn.CrossEntropyLoss(weight = torch.FloatTensor(weights))
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=6)
        self.test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=6)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=6)
        self.test_f1_score = torchmetrics.F1Score(task="multiclass", num_classes=6)
        
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

def train(config):
    data = np.load(config.data_path)
    max_len = data["train_x"].shape[1]
    net = PID_CNN1D(config.num_of_performers, 
                    config.num_of_features,
                    max_len,
                    config.kernal_size,
                    config.dropout,
                    config.dense_size)
                                
    logger = WandbLogger(log_model=True)
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", save_last=True) 
    lr_monitor = LearningRateMonitor(logging_interval="step")
    
    weights = np.unique(data['train_y'],return_counts=True)[1] / data['train_y'].shape[0]
    model = PIDLightningModule(net, config, weights)
    
    logger.watch(model.net)
    
    if config.ckpt_path != None:
        model = model.load_from_checkpoint(config.ckpt_path)
    
    datamodule = DataModule(data, batch_size=config.batch_size)
    
    trainer = pl.Trainer(max_epochs=config.epochs, 
                        logger=logger,
                        accelerator='cpu', 
                        devices=1,
                        precision=16,
                        enable_progress_bar=True,
                        log_every_n_steps=10,
                        callbacks=[checkpoint_callback, lr_monitor])
    
    if config.mode == "train":
        trainer.fit(model, datamodule=datamodule)
        
    trainer.test(model, datamodule=datamodule, ckpt_path=checkpoint_callback.best_model_path)
    

def evaluate(config):
    data = np.load(config.data_path)
    max_len = data["train_x"].shape[1]
    net = PID_CNN1D(config.num_of_performers, 
                    config.num_of_features,
                    max_len,
                    config.kernal_size,
                    config.dropout,
                    config.dense_size)
    
    weights = np.unique(data['train_y'],return_counts=True)[1] / data['train_y'].shape[0]    
    data = data_loader(data['test_x'], data['test_y'], batch_size=config.batch_size)
    
    model = PIDLightningModule.load_from_checkpoint(config.ckpt_path, net=net, config=config, weights=weights)
    
    preds = []
    labels = []
    
    model.eval()
    for batch, label in data:
        inputs = torch.transpose(batch, 1, 2).float()
        with torch.no_grad():
            outputs = model(inputs)
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
    sns.heatmap(pd.DataFrame(text).iloc[:-1, :].T, annot=True, cmap='Greens', yticklabels=PERFROMER + ['accuracy', 'macro avg', 'weighted avg'])
    plt.savefig(os.path.join(config.save_path, "classification_report.png"), bbox_inches='tight')
    
    # Save confustion matrix
    df = pd.DataFrame(np.stack([preds, labels], axis=1), columns=['pred', 'real'])
    df_confusion = pd.crosstab(df['real'], df['pred'], rownames=['Actual'], colnames=['Predicted'],dropna=False,  margins=True)
    plt.clf()
    plt.figure(figsize=(7,5))
    df_confusion = df_confusion.iloc[0:-1,0:-1]/df_confusion.iloc[-1]
    df_confusion = df_confusion.apply(lambda x: round(x, 2))
    df_confusion = df_confusion.iloc[:, 0:-1]
    sns.heatmap(df_confusion, cmap="Blues", annot=True, xticklabels=PERFROMER, yticklabels=PERFROMER)
    plt.title('Confusion Matrix for the Performer Identification')
    plt.savefig(os.path.join(config.save_path, "confusion_matrix.png"), bbox_inches='tight')


def get_args():
    parser = argparse.ArgumentParser(description='')
    
    parser.add_argument("--model", type=str, default="PID_CNN1D")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "evaluate"])
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--num_of_features", type=int, default=len(FEATURES_LIST))
    parser.add_argument("--num_of_performers", type=int, default=6)
    parser.add_argument("--cuda_devices", nargs='+', default=["0"], help="CUDA device ids")
    parser.add_argument("--save_path", type=str, default="evaluation")
    parser.add_argument("--ckpt_path", type=str, default=None)
    args = parser.parse_args()
    parser.print_help()
    
    return args


if __name__ == "__main__":
    args = get_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(args.cuda_devices)
    
    config = {
            'epochs': 1500,
            'batch_size': 16,
            'learning_rate': 8e-5,
            'weight_decay': 1e-7,
            "data_path": "data/processed_data.npz",
            "ckpt_path": args.ckpt_path,
            "save_path": args.save_path,
            "model": "PID_CNN1D",
            "num_of_performers": args.num_of_performers,
            "num_of_features": args.num_of_features,
            "kernal_size": [5, 5, 5, 3],
            "dropout": 0.5,
            "dense_size": 512,
            "mode": args.mode,
        }
    
    if args.mode == "train":
        print("\n------------- Start Training ----------------")
        wandb.init(project="PID-CNN",
            name="pid",
            config=config)
        config = wandb.config
        train(config)
    elif args.mode == "evaluate":
        print("\n------------- Start Evaluating ----------------")
        class Config:
            def __init__(self, config_dict):
                for key, value in config_dict.items():
                    setattr(self, key, value)
        
        config = Config(config)
        evaluate(config)
        
        
        