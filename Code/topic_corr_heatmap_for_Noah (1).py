# heatmap for intra-document topic correlations 

from gensim.models import LdaModel
import sys
import csv
import numpy as np
# from scipy.stats import pearsonr
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
# from sklearn.cluster import DBSCAN

# hyperparameters
num_topics = 50
doc_count = 22691 # NOTE: Based on data_for_R.csv
topics = [1,2,7,8,10,15,16,19,25,27,29,31,37,48] # NOTE: You can play around with this by adding and removing topics to find correlations between topics clusters that emerge as important
corrs = np.identity(len(topics))

# NOTE: update the next line with the list of YOUR topic titles
dictionary = "Framing, Calls for Regulation, Content Compensation, Risks to Market, International, UNCLEAR, UNCLEAR, Emergeing Technologiues, Tech antitrust, UNCLEAR, Telecom Antitrust, UNCLEAR, Trump administation, UNCLEAR, UNCLEAR, Critiques of Capitalism, Competing with..., UNCLEAR,UNCLEAR,Stockmarket, UNCLEAR, Crypto, UNCLEAR,UNCLEAR,UNCLEAR, Censorship, UNCLEAR, EU, growth, US Privacy Regulation, UNCLEAR, Calls for Accoutnability, UNCLEAR, Cloud COmputing, Campaign Critism, Facial Recognition, AI, Bipartisan Support, UNCLEAR, COVID, UNCLEAR,UNCLEAR,UNCLEAR,UNCLEAR,Normative framing, UNCLEAR,UNCLEAR, Interantional, EU Data privacy, Health Care"
dictionary = dictionary.strip().split(",")
topic_names = [dictionary[i] for i in topics]

# extract comment lengths according to the LDA model 
topic_contrib = np.zeros((len(topics),doc_count))
comm_lengths = {}
with open("data_for_R_50.csv","r") as csvfile:
    reader = csv.reader(csvfile)
    for id_,line in enumerate(reader):

        # print progress every 1000 docs
        if id_ != 0 and (id_ % 1000) == 0:
            print(id_)

        if id_ != 0:        
            topic_asgnmnts = eval(line[3])
            topic_counts = Counter(topic_asgnmnts)
            for j in topic_counts:
                if j in topics:
                    topic_contrib[topics.index(j),id_-1] = topic_counts[j] / float(len(topic_asgnmnts))

print(np.mean(np.sum(topic_contrib,axis=0)))

# ldam = LdaModel.load("LDA_50".format(num_topics))

all_corrs = np.corrcoef(topic_contrib) # note that np automatically reorders columns based on the index list

all_df = pd.DataFrame(all_corrs)

sns.set()

sns.heatmap(all_corrs, xticklabels=topic_names, yticklabels=topic_names, annot=True)
sns.heatmap(all_corrs, xticklabels=topic_names, yticklabels=topic_names, annot=True)
plt.xticks(rotation=90) 
plt.show()