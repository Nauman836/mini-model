import torch
import torch.nn.functional as F
import json
from tokenizer import CharTokenizer
from model import MiniGPT

device = "cuda" if torch.cuda.is_available() else "cpu"

with open("config.json", "r") as f:
    config = json.load(f)

block_size = config["block_size"]
n_embd     = config["n_embd"]
n_heads    = config["n_heads"]
n_layers   = config["n_layers"]

text = open("input.txt", encoding="utf-8").read()
tokenizer = CharTokenizer(text)

model = MiniGPT(tokenizer.vocab_size, block_size, n_embd, n_heads, n_layers).to(device)
model.load_state_dict(torch.load("minigpt.pth", map_location=device))
model.eval()

@torch.no_grad()
def generate(model, idx, max_new_tokens=200, temperature=0.8):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.block_size:]
        logits = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        idx = torch.cat([idx, next_token], dim=1)
        if tokenizer.decode([next_token.item()]) == "\n":
            break
    return idx

print("MiniGPT Chat! Type 'quit' to exit.")
chat_history = ""

while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit"]:
        break

    chat_history += f"User: {user_input}\nBot: "
    idx = torch.tensor([[tokenizer.stoi.get(c, tokenizer.stoi[" "]) for c in chat_history]], device=device)
    out_idx = generate(model, idx, max_new_tokens=200, temperature=0.8)
    bot_response = tokenizer.decode(out_idx[0].tolist()[len(idx[0]):])
    print(f"Bot: {bot_response.strip()}")
    chat_history += bot_response.strip() + "\n"
