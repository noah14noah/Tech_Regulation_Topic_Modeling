from gensim.corpora import Dictionary
from gensim.models import LdaModel
import csv
import pandas as pd

num_topics = 50
list_of_texts = []
with open("/Users/njjones14/PycharmProjects/Big_Tech_Regulation/list_of_texts.csv") as f:
    list_of_text_reader = csv.reader(f)
    for row in list_of_text_reader:
        list_of_texts.append(row)

list_of_texts_test = []
with open("/Users/njjones14/PycharmProjects/Big_Tech_Regulation/list_of_texts_test.csv") as f:
    list_of_text_reader = csv.reader(f)
    for row in list_of_text_reader:
        list_of_texts_test.append(row)


dictionary = Dictionary.load("/Users/njjones14/PycharmProjects/Big_Tech_Regulation/dictionary")
dictionary_test = Dictionary.load("/Users/njjones14/PycharmProjects/Big_Tech_Regulation/dictionary_test")

# TODO: needs to be uncommented out after, Train_Models.py is completed
model = LdaModel.load("/Users/njjones14/PycharmProjects/Big_Tech_Regulation/Models/" + str(num_topics) + "/LDA_" + str(num_topics))

corpus = [dictionary.doc2bow(doc) for doc in list_of_texts]
corpus_test = [dictionary_test.doc2bow(doc) for doc in list_of_texts_test]


with open("/Users/njjones14/PycharmProjects/Big_Tech_Regulation/corpus_all_years.csv") as f:  # DATAFRAME
    corpus_cleaned = pd.read_csv(f)

num_sample_comments = 20



