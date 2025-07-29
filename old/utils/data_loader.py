# utils/data_loader.py
import json, ast
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset

class MINDDataset(Dataset):
    def __init__(self, data_dir: str, split: str = "train"):
        base = Path(data_dir) / split
        # behaviors.tsv → history list & impressions (nids+labels)
        self.beh = pd.read_csv(
            base / "behaviors.tsv",
            sep="\t", names=["impr_id","user_id","time","history","impr"],
            usecols=[1,2,3,4], engine="python", quoting=3
        )
        self.beh["history"] = self.beh["history"].fillna("").str.split()
        def parse_imp(x):
            if pd.isna(x): return [],[]
            rs = [p.split("-") for p in x.split()]
            nids, labs = zip(*rs)
            return list(nids), list(map(int,labs))
        tmp = self.beh["impr"].apply(parse_imp).tolist()
        self.beh["impr_nids"], self.beh["impr_lbls"] = zip(*tmp)

        # news.tsv → first 6 cols
        self.news = pd.read_csv(
            base / "news.tsv",
            sep="\t", names=["nid","cat","subcat","title","abs","url","e","r"],
            usecols=[0,1,2,3,4,5], engine="python", quoting=3
        )

        # nid→idx
        self.nid2idx = {n:i for i,n in enumerate(self.news["nid"])}
        # convert lists to idx
        recs = []
        for _, row in self.beh.iterrows():
            h = [self.nid2idx.get(n,0) for n in row["history"]]
            imp = [self.nid2idx.get(n,0) for n in row["impr_nids"]]
            lbl = row["impr_lbls"]
            recs.append((h, imp, lbl))
        self.records = recs

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        h, imp, lbl = self.records[idx]
        return {
            "history":   torch.tensor(h, dtype=torch.long),
            "impr_ids":  torch.tensor(imp, dtype=torch.long),
            "impr_lbls": torch.tensor(lbl, dtype=torch.long),
        }
