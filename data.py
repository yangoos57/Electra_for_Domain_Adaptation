from torch.utils.data import Dataset
import pandas as pd


class book_corpus(Dataset):
    def __init__(self, dir) -> None:
        # data = [token, num, sen]을 가진 Pd.DataFrame
        self.data = pd.read_csv(dir, index_col=0)["sen"].tolist()

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
