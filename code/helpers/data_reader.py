import pandas as pd
import numpy as np

class DataReader:

    DATASET_FOLDER = '/media/marcos/DATA/datasets/preprocessed/'
    DATASET_EXTENSION = '.csv'
    DATASET_SEPARATOR = " "

    def __init__(self, dataset_name: str):
        self.data_path = f'{self.DATASET_FOLDER}{dataset_name}{self.DATASET_EXTENSION}'
    
    def get_dataset(self):
        X = pd.read_csv(self.data_path, sep = self.DATASET_SEPARATOR)
        
        y = None
        
        if 'Y' in X.columns:
            y = X['Y']
            X = X.drop(['Y'], axis=1)

        print('labels: ', max(np.unique(y)))
        print('data: ', X.shape)
        
        return X, y