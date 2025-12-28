import yaml
import custom_model
from typing import Optional
import custom_dataset
from pathlib import Path

def parse_model(config_path: str) -> Optional[custom_model.CustomModel]:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

        input_size = config.get('model').get('input_size')
        layers = config.get('model').get('layers')
        act_fn = config.get('model').get('activation_function')
        act_out = config.get('model').get('output_activation')

        return custom_model.CustomModel(input_size=input_size, layers=layers, act_fn=act_fn, act_out=act_out)
    
    return None

def parse_train_params(config_path: str) -> Optional[dict]:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

        task = config.get('task')
        loss = config.get('train').get('loss')
        optimizer = config.get('train').get('optimizer')
        batch_size = config.get('train').get('batch_size')
        epochs = config.get('train').get('epochs')
        split = config.get('data').get('split')

        return {
            'task': task,
            'loss': loss,
            'optimizer': optimizer,
            'batch_size': batch_size,
            'epochs': epochs,
            'split': split
        }
    
    return None

def parse_experiment_params(config_path: str) -> Optional[dict]:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

        experiments = config.get('experiment')
        return {
            'mode': experiments.get('mode', 'repeat_seeds'),
            'preprocess': experiments.get('preprocess', None),
            'seeds': experiments.get('seeds', [42]),
        }
    
    return None

def parse_dataset_params(config_path: str) -> Optional[custom_dataset.CustomDataset]:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        data = config.get('data')

        dataset_type = data.get('source')

        p = Path(data.get('path'))

        if not p.is_file():
            return None

        target_col = data.get('target_col')
        features_cols = data.get('features_cols')

        if dataset_type == 'csv':
            return custom_dataset.CSVDataset(file_path=p.absolute().as_posix(), 
                                             features_cols=features_cols, 
                                             target_col=target_col)
        
        elif dataset_type == 'sqlite':
            table_name = data.get('sqlite_table_name')
            return custom_dataset.SQLDataset(db_path=p.absolute().as_posix(), 
                                             table_name=table_name, 
                                             features_cols=features_cols, 
                                             target_col=target_col)

    return None

