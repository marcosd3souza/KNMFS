import pandas as pd
import numpy as np

class DataReader:

    DATASET_FOLDER = 'https://github.com/marcosd3souza/FSMethodology/blob/master/train_datasets/'
    DATASET_EXTENSION = '.csv'
    DATASET_SEPARATOR = " "

    def __init__(self, dataset_name: str):
        self.data_path = f'{self.DATASET_FOLDER}{dataset_name}_dataset{self.DATASET_EXTENSION}'
    
    def get_dataset(self):
        X = pd.read_csv(self.data_path, sep = self.DATASET_SEPARATOR)
        
        y = None
        
        if 'Y' in X.columns:
            y = X['Y']
            X = X.drop(['Y'], axis=1)

        print('labels: ', max(np.unique(y)))
        print('data: ', X.shape)
        
        return X, y
