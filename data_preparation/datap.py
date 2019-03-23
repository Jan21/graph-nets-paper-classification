import json
from collections import Counter

# Spojení do jednoho listu
with open('train.jsonl') as f:
    a = []
    for line in f.readlines():
        a.append(line)
with open('dev.jsonl') as f:
    for line in f.readlines():
        a.append(line)

# Vytahnutí id, která se používají
used_ids = []
for line in range(len(a)):
    dic = json.loads(a[line])
    used_ids.append(dic["id"])

# Vytvoření listu jen s hodnotami, které chceme
b = []
for line in range(len(a)):
    dic = json.loads(a[line])
    keys = ["id", "title", "paperAbstract", "outCitations", "venue", "keyPhrases"]
    dicn = {x:dic[x] for x in keys}
    b.append(dicn)


# Vyřazení těch id v outCitation, která nejsou v datasetu

for i in b:
    delete = []
    for j in i["outCitations"]:
        if j not in used_ids:
          delete.append(j)
    for k in delete:
        i["outCitations"].remove(k)

mycounter = Counter()
for i in b:
    mycounter[len(i['outCitations'])] += 1

print(mycounter)

with open('data_fil.jsonl', 'w') as outfile:
    for i in b:
        outfile.write(json.dumps(i) + "\n")

#for i in range(len(b)):
#    for j in b[i]["outCitations"]:
#        if j in used_ids:
#            print("+")
#        else: print("-")

# b[i]["outCitations"].remove(j),
#print(type(a[1]))
#for line in range(0,10):
#    print(a[line])

#dic = json.loads(a[1])
#print(type(dic), dic)

#for key in dic.keys():
#    print(key)
#print(dic["keyPhrases"])
#print(dic["id"])

