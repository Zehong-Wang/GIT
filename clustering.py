import os.path as osp

import numpy as np
import torch
from sklearn.cluster import KMeans

lr = 1e-6
weight = 'none'
vqsize = 64

alignreg = 1.0
vqreg = 0.0

template = osp.join('embeddings', 'lr_{}_weight_{}_alignreg_{}_vqreg_{}_vqsizse_{}.pt')

data = torch.load(template.format(lr, weight, alignreg, vqreg, vqsize))
embeddings = data['embeddings']
labels = data['labels']

clusters = {}
for k in embeddings.keys():
    emb, label = embeddings[k], labels[k]
    kmeans = KMeans(n_clusters=len(np.unique(label)), random_state=42, n_init='auto').fit(emb)

    clusters[k] = kmeans.labels_

torch.save(clusters, osp.join('embeddings', 'clusters',
                              'lr_{}_weight_{}_alignreg_{}_vqreg_{}_vqsizse_{}.pt'.format(lr, weight, alignreg, vqreg,
                                                                                          vqsize)))
