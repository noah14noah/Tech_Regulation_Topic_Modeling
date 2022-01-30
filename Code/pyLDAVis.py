#pyLDAVis visualization
#TODO: THIS IS WORKING 11/18
#Imports
import pyLDAvis.gensim
from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary
import pandas as pd
import csv

num_topics = 30

#Loads
list_of_texts = []
with open("/Users/njjones14/PycharmProjects/Big_Tech_Regulation/list_of_texts_first.csv") as f:
    list_of_text_reader = csv.reader(f)
    for row in list_of_text_reader:
        list_of_texts.append(row)

model = LdaModel.load("/Users/njjones14/PycharmProjects/Big_Tech_Regulation/Models/" + str(num_topics) + "/LDA_" + str(num_topics))
dictionary = Dictionary.load("/Users/njjones14/PycharmProjects/Big_Tech_Regulation/dictionary_first")
corpus = [dictionary.doc2bow(doc) for doc in list_of_texts]

with open("/Users/njjones14/PycharmProjects/Big_Tech_Regulation/corpus_cleaned_first") as f:  # DATAFRAME
    corpus_cleaned = pd.read_csv(f)

vis = pyLDAvis.gensim.prepare(model, corpus, dictionary=dictionary)
pyLDAvis.show(vis)
pyLDAvis.save_json(vis, "/Users/njjones14/PycharmProjects/Big_Tech_Regulation/Models/"+ str(num_topics) + "/pyldavis_fig")