import json

with open('data_fil.jsonl') as f:
    a = []
    for line in f.readlines():
        dic = json.loads(line)
        a.append(dic)

print("Total papers: ", len(a))

badid = []
for i in a:
    if len(i["outCitations"]) == 0:
        badid.append(i["id"])
print("Papres which are not citing: ", len(badid))

goodid = []
for i in a:
    for j in i["outCitations"]:
        if j not in goodid:
            goodid.append(j)

print("Papers which are cited: ", len(goodid))

outid = []
for i in badid:
    if i not in goodid:
        outid.append(i)

print("Papers which are neither cited nor citing: ", len(outid))

b = []

for i in a:
    if i["id"] not in outid:
        b.append(i)
"""
TAKTO NE! zase bych si ubíral tam, kde iteruju...
b = a

for i in b:
    if i["id"] in outid:
        b.remove(i)
"""
print("Papers to be included: ", len(b))

with open('data_filtered.jsonl', 'w') as outfile:
    for i in b:
        outfile.write(json.dumps(i) + "\n")