import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class CreditDataset:
    def __init__(self,fitM, path="./data/credit_train.csv"):
        self.fitM = fitM
        self.path = path
        self.data = pd.read_csv(self.path).to_numpy()
        # choose gender as our sensitive attribute
        self.total_feats = 38
        self.t_feat = 0
        self.bin_feats = list(range(0,34))
        self.con_feats = [34, 35, 36]
        self.x_feats = [i+1 for i in self.bin_feats+self.con_feats]
        self.target_idx = 38

        # fit batch size
        m = self.data.shape[0] // fitM
        N = m*fitM

        self.data_t = self.data[:N, self.t_feat].reshape(-1,1).astype('float32')
        self.data_x = self.data[:N, self.x_feats].astype('float32')
        self.data_y = self.data[:N, self.target_idx].reshape(-1,1).astype('float32')
    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        t = self.data_t[idx, :]
        x = self.data_x[idx, :]
        y = self.data_y[idx, :]
        return t, x, y

# for testing
if __name__ == "__main__":
    trainset = CreditDataset(fitM=64)
    print(trainset.data_x.shape[0]/64)