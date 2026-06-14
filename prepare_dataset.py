import os

corpus_dir = "archive"  # path to extracted Cornell Dialogs
lines_file = os.path.join(corpus_dir, "movie_lines.txt")

lines = []
with open(lines_file, "r", encoding="iso-8859-1") as f:
    for line in f:
        parts = line.strip().split(" +++$+++ ")
        if len(parts) == 5:
            lines.append(parts[4])

lines = [line.replace("\t"," ").replace("\n"," ").strip() for line in lines if line.strip()]

with open("input.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"Prepared dataset with {len(lines)} lines as input.txt")
