import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from lxml import etree
import math

class WikiArticlesDataset(Dataset):
    def __init__(self, getDataForText, file_path, chunk_size):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.getDataForText = getDataForText

        with open(file_path, "rb") as f: root = etree.XML(f.read())
        page_count = int(root.xpath("count(//pages/page)"))
        print('page_count', page_count)
        page_count = 20
        self.total_chunks = math.ceil(page_count / chunk_size)
        print('total chunks', self.total_chunks)

    def __len__(self):
        return self.total_chunks

    def __getitem__(self, chunk_idx):
        print('chunk idx is', chunk_idx)

        start_pos = chunk_idx * self.chunk_size
        end_pos = start_pos + self.chunk_size

        print(f"Loading data for {start_pos} to {end_pos - 1}")

        xpath = f"//pages/page[position() >= {start_pos} and not(position() >= {end_pos})]/revision/text"
        articles = pd.read_xml('./data/enwik9.xml', xpath=xpath)
        
        # remove redirect stubs 
        articles = articles[articles['text'].str.match(r'^( )*#redirect', case=False) == False].reset_index()[['text']]

        data = articles['text'].swifter.progress_bar(False).apply(self.getDataForText)

        flat_data = data.explode().dropna()

        flat_data = flat_data[:10000]

        target_tensor = torch.LongTensor([d[0] for d in flat_data])
        input_tensor = torch.LongTensor([d[1] for d in flat_data])

        return target_tensor, input_tensor
    
    def collate_fn(self, batch):
        """
        Custom collate function to batch the skipgram pairs
        """
        targets, contexts = zip(*batch)
        return torch.stack(targets), torch.stack(contexts)

