import json
import matplotlib.pyplot as plt
import gensim
import numpy as np
import spacy



with open('data_connected_graph.jsonl') as f:
    a = []
    for line in f.readlines():
        dic = json.loads(line)
        a.append(dic)
