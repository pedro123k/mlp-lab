from torch.utils.data import Dataset
import pandas as pd
from typing import Optional, Callable, List, Tuple
import torch 
import sqlite3
from abc import ABC, abstractmethod
import numpy as np

class CustomDataset(Dataset, ABC):
    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

class CSVDataset(CustomDataset):
    def __init__(self, file_path: str, features_cols: List[int], target_col: int , transform: Optional[Callable] = None) -> None:
        self._data = pd.read_csv(file_path)
        self._transform = transform
        self._features_cols = features_cols
        self._target_col = target_col
        self._avgs = torch.from_numpy(self._data.to_numpy()).float().mean(axis=0)
        self._vars = torch.from_numpy(self._data.to_numpy()).float().var(axis=0)
        self._mins = torch.from_numpy(self._data.to_numpy()).float().min(axis=0)
        self._maxs = torch.from_numpy(self._data.to_numpy()).float().max(axis=0)

    def __len__(self) -> int:
        return len(self._data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        features = torch.from_numpy(self._data.iloc[idx, self._features_cols].values).float()
        target = torch.from_numpy(np.array(self._data.iloc[idx, self._target_col])).float()


        if self._transform:
            features = self._transform(features, [self._avgs, self._vars, self._mins, self._maxs])

        return features, target
    
class SQLDataset(CustomDataset):
    def __init__(self, db_path: str, table_name: str, features_cols: List[str], target_col: str, transform: Optional[Callable] = None) -> None:
        self._conn = sqlite3.connect(db_path)
        self._cursor = self._conn.cursor()
        self._table_name = table_name
        self._transform = transform
        self._features_cols = features_cols
        self._target_col = target_col

        self._cursor.execute(f"SELECT COUNT(*) FROM {self._table_name}")
        self._length = self._cursor.fetchone()[0]

        self._cursor.execute(f"SELECT {','.join(f'AVG({f})' for f in features_cols)} FROM {self._table_name}")
        self._avgs = torch.tensor(self._cursor.fetchone(), dtype=torch.float32)

        self._cursor.execute(f"SELECT {','.join(f'AVG({f}*{f}) - AVG({f})*AVG({f})' for f in features_cols)} FROM {self._table_name}")
        self._vars = torch.tensor(self._cursor.fetchone(), dtype=torch.float32)

        self._cursor.execute(f"SELECT {','.join(f'MIN({f})' for f in features_cols)} FROM {self._table_name}")
        self._mins = torch.tensor(self._cursor.fetchone(), dtype=torch.float32)

        self._cursor.execute(f"SELECT {','.join(f'MAX({f})' for f in features_cols)} FROM {self._table_name}")
        self._maxs = torch.tensor(self._cursor.fetchone(), dtype=torch.float32)

    def __len__(self) -> int:
        return self._length
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        query = f"SELECT {', '.join(self._features_cols)}, {self._target_col} FROM {self._table_name} LIMIT 1 OFFSET {idx}"
        self._cursor.execute(query)
        row = self._cursor.fetchone()

        if row is None:
            raise ValueError(f"No row found for idx={idx}. Query: {query}")

        features = torch.tensor(row[:-1], dtype=torch.float32)
        target = torch.tensor([row[-1]], dtype=torch.float32)

        if self._transform:
            features = self._transform(features, [self._avgs, self._vars, self._mins, self._maxs])

        return features, target
    
    @property
    def transform(self) -> Optional[Callable]:
        return self._transform
    
    @transform.setter
    def transform(self, transformantion: Optional[Callable]) -> None:
        self._transform = transformantion
    
    def __del__(self):
        if hasattr(self, "_conn"):
            self._conn.close()

