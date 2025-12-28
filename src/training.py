from src.custom_dataset import CustomDataset
from src.custom_model import CustomModel
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
import math
import time
from tqdm import tqdm

class Traning:
    def __init__(self, model: CustomModel, dataset: CustomDataset, train_params: dict, device: torch.device) -> None:
        self._model = model
        self._model.to(device)
        self._dataset = dataset
        self._train_params = train_params
        self._device = device
        self._history = {
            'train_loss': np.array([]),
            'val_loss': np.array([]),
            'eval_metrics': np.array([])
        }

    def _make_consistency(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        if t1.dim() != t2.dim():
            return t2.view_as(t1)
        return t2
            

    def train_model(self) -> None:
        loss = nn.MSELoss()

        if self._train_params['loss'] == 'bce':
            loss = nn.BCELoss()
        elif self._train_params['loss'] == 'bce_logits':
            loss = nn.BCEWithLogitsLoss()

        opt = torch.optim.Adam(self._model.parameters(), lr=0.001)
        lr = self._train_params['optimizer'].get('lr', 0.001)

        if self._train_params['optimizer'].get('name', 'adam') == 'adam':
            betas = self._train_params['optimizer'].get('betas', (0.9, 0.999))
            opt = torch.optim.Adam(self._model.parameters(), lr=lr, betas=betas)
        elif self._train_params['optimizer'].get('name', 'adam') == 'sgd':
            opt = torch.optim.SGD(self._model.parameters(), lr=lr)
        
        batch_size =  self._train_params['batch_size']
        split = self._train_params['split'].get('test_size', (0.7, 0.2, 0.1))
        shuffle = self._train_params['split'].get('shuffle', True)

        ds_train, ds_val, ds_test = random_split(self._dataset, split)

        dataloader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=shuffle)
        dataloader_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
        dataloader_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

        t_start = time.perf_counter()

        pbar = tqdm(total=self._train_params['epochs'], desc="Training Progress")

        for epoch in range(self._train_params['epochs']):
            loss_agg = 0.0

            for x, y in dataloader_train:
                x, y = x.to(self._device), y.to(self._device)
                opt.zero_grad()
                outputs = self._model(x)
                y = self._make_consistency(outputs, y)        
                l = loss(outputs, y)
                l.backward()
                opt.step()
                loss_agg += l.detach().cpu().item()

            loss_agg /= len(dataloader_train)

            self._model.eval()
            with torch.no_grad():
                val_loss_agg = 0.0
                for x_val, y_val in dataloader_val:
                    x_val, y_val = x_val.to(self._device), y_val.to(self._device)
                    outputs_val = self._model(x_val)
                    y_val = self._make_consistency(outputs_val, y_val)
                    l_val = loss(outputs_val, y_val)
                    val_loss_agg += l_val.detach().cpu().item()

                val_loss_agg /= len(dataloader_val)

            pbar.set_description(f"Epoch {epoch+1}/{self._train_params['epochs']}")
            pbar.set_postfix(train_loss=loss_agg, val_loss=val_loss_agg)
            pbar.update(1)
            
            self._history['train_loss'] = np.append(self._history['train_loss'], loss_agg)
            self._history['val_loss'] = np.append(self._history['val_loss'], val_loss_agg)

        self._model.eval()
        type_task = self._train_params['task']
        
        metric_agg = 0.0
        n = 0
        metric_aux = 0.0
        metric_aux2 = 0.0 
        with torch.no_grad():
            for x_test, y_test in dataloader_test:
                x_test, y_test = x_test.to(self._device), y_test.to(self._device)
                outputs_test = self._model(x_test)
                y_test = self._make_consistency(outputs_test, y_test)
                    
                if type_task == 'classification': 
                    metric_value = (y_test == (outputs_test > 0.5).float()).float()
                    metric_agg += metric_value.sum().detach().cpu().item()
                elif type_task == 'regression':
                    metric_value = (y_test - outputs_test) ** 2
                    metric_aux += outputs_test.sum().detach().cpu().item()
                    metric_aux2 += (outputs_test ** 2).sum().detach().cpu().item()
                    metric_agg += metric_value.sum().detach().cpu().item()
                else:
                    raise ValueError(f"Unknown task type: {type_task}")
                n += y_test.numel()


        metric_value = metric_agg / n
        
        if type_task == 'regression':
            ss1 = n * metric_agg
            ss2 = metric_aux2 - (metric_aux ** 2) / n

            eps = 1e-8

            if math.isnan(ss2) or abs(ss2) < eps:
                metric_value = 0.0
            else:
                metric_value = 1 - ss1 / ss2


        self._history['eval_metrics'] = metric_value

        t_end = time.perf_counter()
        t_elapsed = t_end - t_start
        self._history['training_time_total'] = t_elapsed
        self._history['training_time_per_epoch'] = t_elapsed / self._train_params['epochs']

        pbar.close()

        print(f"Training Finished: Evaluation Metric ({self._train_params['task']}): {metric_value:.4f}")

    @property
    def history(self) -> dict:
        return self._history