import pandas as pd
from defaults import model, corpus, num_topics, corpus_cleaned, num_sample_comments, dictionary
from gensim.models import ldamodel
import csv


def Convert(tup, di):
    for a, b in tup:
        di.setdefault(a, []).append(b)
    return di


# initialize dataframe
sample_comments = pd.DataFrame(columns=["ID", "document", "topic_probabilities"])

# this for loop appends dictionary of topic cotributions per document data into dataframe

for idx, row in corpus_cleaned.iterrows():
    input = row["Content"].strip().split()
    # TODO: This might not be correct, this might have to be training, but I want to use cleaned_corpus, to identify possible more representive comments
    bow = dictionary.doc2bow(input)  # TODO: since I am using corpus_cleaned, I do not have bigrams, could be a source of unreliable topics
    topic_props = model.get_document_topics(bow)
    d = {}
    dict_topics_props = Convert(topic_props, d)
    r = {"ID": row["ID"], "document": row["Content"], "topic_probabilities": dict_topics_props}
    print(r["topic_probabilities"])
    sample_comments = sample_comments.append(r, ignore_index=True)


total_dict = {} #will be dictionary of length num_topics with tuples of (ID, topic contribution (for that topic))

# for row in sample_comments.iterrows():
#     for topic in row["topic_probabilities"].keys():
#         if topic in total_dict.keys():
#             tup = (row["ID"], )
#
#

k_cols = range(num_topics)
output_dict = {}
for i in k_cols:
    listing = []
    for idx, row in sample_comments.iterrows():
        if i in row["topic_probabilities"].keys():
            tup = (row["ID"], row["topic_probabilities"][i])
            listing.append(tup)
    # at this point, for a single topic we should have all of the IDs of documets that cotribute to topic, and the
    # contributions of that document

    # now sort and select
    sort = sorted(listing, reverse=True, key=lambda x: x[1])
    selected = sort[:num_sample_comments]
    output_dict[i] = selected

print(output_dict)

with open("/Users/njjones14/PycharmProjects/Big_Tech_Regulation/Models/" + str(num_topics) + "/sample_comments", "w") as f:
    for i in output_dict:
        print("Topic: ", str(i), file=f)
        print("\n")
        for j in output_dict[i]:
            print(str(j), file=f)
            print("\n")


