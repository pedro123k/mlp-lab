import sys
from pathlib import Path
import src.parses as parses
import src.logger as logger
import src.training as training
import torch
import time
from typing import Optional

def run_lab(config_path: Path, outputs_path: Path, labelname: Optional[str]) -> None:
    
    print("Parsing configuration file...")

    print("Parsing data configuration...")
    dataset = parses.parse_dataset_params(config_path.absolute().as_posix())

    if dataset is None:
        raise ValueError("Dataset could not be parsed.")
    
    print("Parsing training configuration...")
    train_params = parses.parse_train_params(config_path.absolute().as_posix())

    if train_params is None:
        raise ValueError("Training parameters could not be parsed.")
        
    exp_params = parses.parse_experiment_params(config_path.absolute().as_posix())

    if exp_params is None:
        raise ValueError("Experiment parameters could not be parsed.")
    
    print("Starting repeat_seeds mode...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f'Device={device}')
    
    if exp_params['mode'] == 'repeat_seeds':
        print("Starting preprocess_grid mode...")

        seeds = exp_params['seeds']

        if seeds is None or type(seeds) is not list or len(seeds) == 0:
            raise ValueError("In repeat_seeds mode, 'seeds' must be a non-empty list of integers.")

        log = logger.Logger(mode='repeat_seeds', task=train_params.get('task'))

        for seed in seeds:
            print(f"Starting training with seed: {seed}...")

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

            model = parses.parse_model(config_path.absolute().as_posix())

            if model is None:
                raise ValueError("Model could not be parsed.")

            trainer = training.Traning(model=model, dataset=dataset, train_params=train_params, device=device)

            trainer.train_model()
            trainer.history['seed'] = seed


            log.add_history(str(time.time_ns()), trainer.history)

        label = labelname or str(time.time_ns())

        log.log_runs((outputs_path / f"{label}_runs.csv").absolute().as_posix())
        log.log_summary((outputs_path / f"{label}_summary.json").absolute().as_posix())
        log.log_graphs((outputs_path / f"{label}_graph.png").absolute().as_posix())
        

    elif exp_params['mode'] == 'preprocess_grid':
        print("Starting preprocess_grid mode...")

        preprocesses = exp_params['preprocess']

        if type(preprocesses) is not list and type(preprocesses) is not str and preprocesses is not None:
            raise ValueError("In preprocess_grid mode, 'preprocess' must be null, a string or a list of strings.")

        if type(preprocesses) is not list:
            preprocesses = [preprocesses]

        

        seed = exp_params.get('seed', 42)
        if type(seed) is list:
                raise ValueError("In preprocess_grid mode, 'seed' should be a single integer value.")
        
        log = logger.Logger(mode='preprocess_grid', task=train_params.get('task'), global_seed=seed)

        for preprocess in preprocesses:
            print(f"Preprocessing with: {preprocess} [seed={seed}]...")
            
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

            if preprocess is None:
                dataset.transform = None
            elif preprocess == 'minmax':
                dataset.transform = lambda x, stats: (x - stats[2]) / (stats[3] - stats[2])
            elif preprocess == 'standard':
                dataset.transform = lambda x, stats: (x - stats[0]) / torch.sqrt(stats[1])
            else:
                raise ValueError(f"Unknown preprocess method: {preprocess}")
        
            model = parses.parse_model(config_path.absolute().as_posix())

            if model is None:
                raise ValueError("Model could not be parsed.")

            trainer = training.Traning(model=model, dataset=dataset, train_params=train_params, device=device)

            trainer.train_model()
            trainer.history['preprocess'] = preprocess

            log.add_history(str(time.time_ns()), trainer.history)

        label = labelname or str(time.time_ns())

        log.log_runs((outputs_path / f"{label}_runs.csv").absolute().as_posix())
        log.log_summary((outputs_path / f"{label}_summary.json").absolute().as_posix())
        log.log_graphs((outputs_path / f"{label}_graph.png").absolute().as_posix())
        
    else:
        raise ValueError(f"Unknown experiment mode: {exp_params['mode']}")
    
    print("Done!")

def init() -> None:
    cmd_list = sys.argv

    config_path = None
    output_path = Path("./results")
    labelname = None
    
    for cmd in cmd_list:
        if '=' in cmd:
            key, value = cmd.split('=', 1)
            if key == '--config':
                config_path = Path(value)

                if not Path.is_file(config_path):
                    raise FileNotFoundError(f"Configuration file not found at: {value}")
                
            elif key == '--outdir':
                output_path = Path(value)
                Path.mkdir(output_path, parents=True, exist_ok=True)

            elif key == '--label':
                labelname = value
    
    run_lab(config_path, output_path, labelname)

if __name__ == "__main__":
    init()