import torch

class CharDataset:
    def __init__(self, text, tokenizer, block_size):
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    def get_batch(self, batch_size, device):
        ix = torch.randint(0, len(self.data)-self.block_size-1, (batch_size,))
        x = torch.stack([self.data[i:i+self.block_size] for i in ix])
        y = torch.stack([self.data[i+1:i+self.block_size+1] for i in ix])
        return x.to(device), y.to(device)
