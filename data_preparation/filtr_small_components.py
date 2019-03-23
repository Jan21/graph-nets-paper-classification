import json
import pickle

with open('data_filtered.jsonl') as f:
    a = []
    for line in f.readlines():
        dic = json.loads(line)
        a.append(dic)

with open("used_indices.txt", "rb") as f:
    used_indices = pickle.load(f)

data = []
for i in a:
    if i["id"] in used_indices:
        data.append(i)

print(len(data))

with open('data_connected_graph.jsonl', 'w') as outfile:
    for i in data:
        outfile.write(json.dumps(i) + "\n")