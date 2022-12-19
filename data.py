from torch.utils.data import Dataset
import pandas as pd


class book_corpus(Dataset):
    def __init__(self, dir) -> None:
        self.data = pd.read_csv(dir)["0"].tolist()

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
