from torch.utils.data import Dataset, Sampler
import logging

logger = logging.getLogger(__name__)

def subset_extraction(df, query_col, document_col, mix_languages = False):
    # if mix languages is True, query col becomes a prefix
    if mix_languages:
        return ExpandedDataset(df, anchor_col_prefix=query_col, doc_col=document_col)
    else:
        if query_col not in df.columns or document_col not in df.columns:
            logger.error(f"Requested {query_col} or {document_col} not found in dataset.")
            return
        
        return BasicDataset(df[[query_col, document_col]].rename(columns={
            query_col: "anchor",
            document_col: "doc"
        }))


class BasicDataset(Dataset):
    def __init__(self, df):
        self.df = df.sample(frac=1).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx]


class ExpandedDataset(Dataset):
    def __init__(self, df, anchor_col_prefix, doc_col):
        df = df[[f"{anchor_col_prefix}_en", f"{anchor_col_prefix}_fr", doc_col]]

        self.df_en = df.sample(frac=1)[[f"{anchor_col_prefix}_en", doc_col]].rename(columns={f"{anchor_col_prefix}_en": "anchor", doc_col: "doc"}).reset_index(drop=True)
        self.df_fr = df.sample(frac=1)[[f"{anchor_col_prefix}_fr", doc_col]].rename(columns={f"{anchor_col_prefix}_fr": "anchor", doc_col: "doc"}).reset_index(drop=True)

        self.len_en = len(self.df_en)
        self.len_fr = len(self.df_fr)

    def __len__(self):
        return self.len_en + self.len_fr

    def __getitem__(self, idx):
        # return en rows first, then fr rows
        if idx < self.len_en:
            row = self.df_en.iloc[idx]
        else:
            row = self.df_fr.iloc[idx - self.len_en]
        return row



class InterleaveBatchSampler(Sampler):
    def __init__(self, lengths, batch_size, mode="sequential"):
        """
        lengths: list of dataset lengths (e.g., [len_en] or [len_en, len_fr])
        batch_size: chunk size per dataset (NOT DataLoader batch size)
        mode: "sequential" or "interleave"
        shuffle: whether to shuffle within each dataset
        """
        self.lengths = lengths
        self.batch_size = batch_size
        self.mode = mode

        # Build index groups
        self.index_groups = []
        start = 0
        for l in lengths:
            indices = list(range(start, start + l))
            self.index_groups.append(indices)
            start += l

    def __iter__(self):
        # Single dataset: plain sequential sampling
        if len(self.index_groups) == 1:
            yield from self.index_groups[0]
            return

        # two datasets: either sequential (EN then FR) or interleaved (batch of EN, then batch of FR, etc.)

        if self.mode == "sequential":
            # all EN then all FR
            for g in self.index_groups:
                yield from g
            return

        elif self.mode == "interleave":
            # interleave batches from each dataset
            max_len = max(len(g) for g in self.index_groups)
            i = 0

            while i < max_len:
                for g in self.index_groups:
                    if i < len(g):
                        yield from g[i:i+self.batch_size]
                i += self.batch_size

    def __len__(self):
        return sum(self.lengths)



