import matplotlib.pyplot as plt
import math
import csv
import json
import numpy as np
from typing import Optional

class Logger:
    def __init__(self, task: Optional[str]=None, mode: Optional[str]=None, global_seed: Optional[str]=None) -> None:
        self._hists = {}
        self._global_seed = global_seed
        self._task = task
        self._mode = mode

    @property
    def seed(self) -> Optional[int]:
        return self._global_seed
    
    @seed.setter
    def seed(self, seed: int) -> None:
        self._global_seed = seed

    @property
    def task(self) -> Optional[str]:
        return self._task
    
    @task.setter
    def task(self, task: str) -> None:
        self._task = task

    @property
    def mode(self) -> Optional[str]:
        return self._mode
    
    @mode.setter
    def mode(self, mode: str) -> None:
        self._mode = mode

    def add_history(self, train: str, hist: dict) -> None:
        self._hists[train] = hist


    def log_graphs(self, save_path: str) -> None:
        num_graphs = len(self._hists)
        n_cols = min(3, num_graphs)
        n_rows = math.ceil(num_graphs / n_cols)

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axs = np.array(axs).reshape(-1)  # garante que axs seja 1D

        for idx, ax in enumerate(axs[:num_graphs]):
            train_name = list(self._hists.keys())[idx]
            history = self._hists[train_name]

            ax.plot(history['train_loss'], label='Train Loss', color='blue')
            ax.plot(history['val_loss'], label='Val Loss', color='orange')
            ax.set_title(f'Training History: {train_name}', fontsize=10)
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True)

        # Remove subplots extras se houver
        for ax in axs[num_graphs:]:
            ax.axis('off')

        plt.tight_layout()
        fig.savefig(save_path, dpi=300)
        plt.close(fig)

        print("Plots saved to", save_path)


    def log_runs(self, save_path: str) -> None:
        with open(save_path, mode='w', encoding='utf-8', newline='') as csvfile:
            
            headers = ['Training', 'Final Train Loss', 'Final Val Loss', 'Evaluation Metric', 'Total Time', 'Time per Epoch']
            writer = csv.writer(csvfile)

            if self._task == 'classification':
                headers[3] = 'Accuracy'
            elif self._task == 'regression':
                headers[3] = 'R2'

            if self._mode == 'preprocess_grid':
                headers.insert(1, 'Preprocess')
            elif self._mode == 'repeat_seeds':
                headers.insert(1, 'Seed')

            writer.writerow(headers)

            for train_name, history in self._hists.items():
                final_train_loss = history['train_loss'][-1] if len(history['train_loss']) > 0 else 'N/A'
                final_val_loss = history['val_loss'][-1] if len(history['val_loss']) > 0 else 'N/A'
                eval_metric = history['eval_metrics']
                total_time = f"{history.get('training_time_total', 'N/A'):.4f}"
                time_per_epoch = f"{history.get('training_time_per_epoch', 'N/A'):.4f}"

                if self._mode == 'preprocess_grid':
                    writer.writerow([train_name,(history['preprocess'] or "none") ,final_train_loss, final_val_loss, eval_metric, total_time, time_per_epoch])
                elif self._mode == 'repeat_seeds':
                    writer.writerow([train_name, history['seed'] ,final_train_loss, final_val_loss, eval_metric, total_time, time_per_epoch])

            print ("Runs logged to", save_path)

    def log_summary(self, save_path: str) -> None: 
        with open(save_path, mode="w", encoding="utf-8") as jsonfile:
            summary = {}
            
            summary['num_runs'] = len(self._hists)
            summary['task'] = self._task
            summary['mode'] = self._mode
            
            if self._global_seed is not None:
                summary['seed'] = self._global_seed

            best_run = list(self._hists.keys())[0]
            worst_run= list(self._hists.keys())[0]

            agg_time = np.array([])
            agg_metrics = np.array([])

            for run_name, history in self._hists.items():
                agg_time = np.append(agg_time, history['training_time_total'])
                agg_metrics = np.append(agg_metrics, history['eval_metrics'])

                if history['eval_metrics'] > self._hists[best_run]['eval_metrics']:
                    best_run = run_name

                if history['eval_metrics'] < self._hists[worst_run]['eval_metrics']:
                    worst_run = run_name

            summary['time_avg'] = np.mean(agg_time).item()
            summary['time_std'] = np.std(agg_time).item()

            summary["metrics_avg"] = np.mean(agg_metrics).item()
            summary["metrics_std"] = np.std(agg_metrics).item()

            summary['best_run'] = best_run
            summary['best_run_metric'] = self._hists[best_run]['eval_metrics']
            summary['best_run_time'] = self._hists[best_run]['training_time_total']

            summary['worst_run'] = worst_run
            summary['worst_run_metric'] = self._hists[worst_run]['eval_metrics']
            summary['worst_run_time'] = self._hists[worst_run]['training_time_total']

            json.dump(summary, jsonfile, indent=4)

            print("Summary saved to", save_path)



        