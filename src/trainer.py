import os
import pickle
import shutil
import numpy as np
import pandas as pd
import scipy as sp
import torch.nn as nn
import torch

from src.dataset import TabDataModule
from src.utils.metrics import compute_metrics, dump_metrics
from src.utils.other import apply_model


class Trainer:
    def __init__(
            self,
            model: nn.Module,
            datamodule: TabDataModule,
            optimizer: torch.optim.Optimizer,
            loss_fn: nn.Module,
            run_dir: str
    ):
        self.model = model
        self.datamodule = datamodule
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.run_dir = run_dir

        self.checkpoints_dir = os.path.join(self.run_dir, 'checkpoints')

    def train(self, n_epochs: int, batch_size: int, report_frequency: int):
        self._create_checkpoints_dir()
        self._save_split()

        train_dataloader = self.datamodule.get_dataloader('train', batch_size, shuffle=True)
        for epoch in range(1, n_epochs + 1):
            self.model.train()

            epoch_losses = []
            for iteration, batch in enumerate(train_dataloader):
                self.optimizer.zero_grad()

                x_batch, y_batch = batch
                
                predict = apply_model(self.model, *x_batch)
                predict = predict.squeeze(1)

                loss = self.loss_fn(predict, y_batch)
                loss.backward()

                epoch_losses.append(loss.detach().cpu().numpy())
                
                self.optimizer.step()
                
                # if iteration % report_frequency == 0:
                #     print(f'(epoch) {epoch:3d} (iteration) {iteration:5d} (loss) {loss.item():.4f}')

            train_metrics, train_loss = self._evaluate('train', batch_size)
            val_metrics, val_loss = self._evaluate('val', batch_size)

            print(
                f'Epoch {epoch:03d} | '
                f'Train auroc {train_metrics["auroc"]:.4f}, auprc: {train_metrics["auprc"]:.4f} | '
                f'Val auroc {val_metrics["auroc"]:.4f}, auprc: {val_metrics["auprc"]:.4f} | '
                f'Loss train {train_loss:.4f}, val {val_loss:.4f}'
            )
            
            self._save_checkpoint(self.model, epoch, self.checkpoints_dir)
            self._save_metrics(
                {'train': train_metrics, 'val': val_metrics},
                {'train': train_loss, 'val': val_loss},
                epoch
            )
            self._save_loss(epoch_losses)

    @torch.no_grad()
    def _evaluate(self, part_name: str, batch_size: int):
        self.model.eval()

        dataloader = self.datamodule.get_dataloader(part_name, batch_size, shuffle=False)

        predict = []
        target = []
        for batch in dataloader:
            x_batch, y_batch = batch

            predict.append(apply_model(self.model, *x_batch))
            target.append(y_batch)

        predict = torch.cat(predict).squeeze(1).cpu().numpy()
        predict = np.round(sp.special.expit(predict))

        target = torch.cat(target).cpu().numpy()

        loss = float(self.loss_fn(torch.tensor(predict), torch.tensor(target)).cpu())
        metrics = compute_metrics(predict, target)

        return metrics, loss
    
    def _save_split(self):
        save_path = os.path.join(self.run_dir, 'datamodule.pickle')

        self.datamodule.split.save(save_path)
    
    def _save_loss(self, epoch_losses: list):
        save_path = os.path.join(self.run_dir, 'loss.npy')

        current_losses = np.array(epoch_losses)

        if not os.path.exists(save_path):
            np.save(save_path, current_losses)
        else:
            saved_losses = np.load(save_path)
            new_losses = np.concatenate((saved_losses, current_losses))

            np.save(save_path, new_losses)

    def _save_metrics(self, metrics, loss, epoch: int):
        save_path = os.path.join(self.run_dir, 'train_val_metrics.csv')

        current_metrics = pd.DataFrame.from_dict({
            'epoch': [epoch],
            'train_auprc': [metrics['train']['auprc']],
            'val_auprc': [metrics['val']['auprc']],
            'train_auroc': [metrics['train']['auroc']],
            'val_auroc': [metrics['val']['auroc']],
            'train_loss': [loss['train']],
            'val_loss': [loss['val']],
        })

        if not os.path.exists(save_path):
            current_metrics.to_csv(save_path, index=False)
        else:
            saved_metrics = pd.read_csv(save_path)
            new_metrics = pd.concat((saved_metrics, current_metrics))

            new_metrics.to_csv(save_path, index=False)

    def _create_checkpoints_dir(self):
        if os.path.exists(self.checkpoints_dir):
            shutil.rmtree(self.checkpoints_dir)
        os.makedirs(self.checkpoints_dir)

    def _save_checkpoint(
            self,
            model: nn.Module,
            epoch: int,
            experiment_dir: str,
    ):  
        checkpoint_path = os.path.join(experiment_dir, f'{str(epoch).zfill(4)}.pt')
        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'epoch': epoch
            },
            checkpoint_path
        )
        
        