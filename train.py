import torch
import torch.nn as nn
import json
from tokenizer import CharTokenizer
from dataset import CharDataset
from model import MiniGPT

with open("config.json", "r") as f:
    config = json.load(f)

block_size    = config["block_size"]
n_embd        = config["n_embd"]
n_heads       = config["n_heads"]
n_layers      = config["n_layers"]
batch_size    = config["batch_size"]
learning_rate = config["learning_rate"]
max_steps     = config["max_steps"]
eval_interval = config["eval_interval"]

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

text = open("input.txt", encoding="utf-8").read()
tokenizer = CharTokenizer(text)
dataset = CharDataset(text, tokenizer, block_size)

model = MiniGPT(tokenizer.vocab_size, block_size, n_embd, n_heads, n_layers).to(device)
print(f"Total parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

model.train()
for step in range(1, max_steps+1):
    x, y = dataset.get_batch(batch_size, device)
    logits = model(x)
    B,T,V = logits.shape
    loss = criterion(logits.view(B*T,V), y.view(B*T))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % eval_interval == 0:
        print(f"Step {step}/{max_steps} | Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "minigpt.pth")
print("Model saved as minigpt.pth")
