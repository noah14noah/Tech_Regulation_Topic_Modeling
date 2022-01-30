import gensim
import ast
import sys
import numpy
from defaults import model, dictionary

num_topics = 25
ENTIRE_CORPUS = True
## Load model, dictionary and set to top topics

model.get_model()
dictionary = gensim.corpora.Dictionary.load(dictionary)

category = ""
top_topics = [16,20,23]


## Extract the top words of top topics
# NOTE: Requires top_words_all_[num_topics] in the model's directory.
# Please manually copy from [output_path]
top_word = {key:0 for key in top_topics}
#with open("LDA_full-corpus:{}_{}/top_words_all_{}".format(ENTIRE_CORPUS,num_topics,num_topics),"r") as f:
with open("/Users/njjones14/PycharmProjects/Big_Tech_Regulation/Models/" + str(num_topics) + "top_words_all_" + str(num_topics), "r") as f:
    for idx,line in enumerate(f):
        if idx == 100:
            break
        if idx != 0 and idx % 2 != 0:
            if int(idx/2) in top_topics:
                top_word[int(idx / 2)] = (ast.literal_eval(line)) # comment out the next
            # line if you want [topn] top words, not [topn]/2
            # top_word[int(idx / 2)] = top_word[int(idx / 2)][:int((len(top_word[int(idx / 2)]) / 2 )- 1)]

## Identify unique and almost unique top words of top topics
uniques = {key:[] for key in top_word.keys()}
twos = {key:[] for key in top_word.keys()}
other_top_words = {key:[] for key in top_word.keys()}
for topic in top_word.keys():
    if topic in top_topics:
        print (topic)
        for other_topic in top_word.keys():
            if other_topic in top_topics:
                print( other_topic)
                print( other_topic == topic)
                if other_topic != topic:
                    for word in [x[0] for x in top_word[other_topic]]:
                        other_top_words[topic].append(word)
# for topic in top_word.keys():
#     for word in top_word[topic]:
#
#         if word not in other_top_words[topic]:
#             uniques[topic].append(word)

counts = {}

for topic in top_word.keys():
    for word in top_word[topic]:
        if word[0] in counts:
            counts[word[0]] += 1
        else:
            counts[word[0]] = 1

for topic in top_word.keys():
    for word in top_word[topic]:# top_word[idx / 2] = top_word[idx / 2][:5] # top 6
        if counts[word[0]] == 1:
            uniques[topic].append(word[0])
        elif counts[word[0]] == 2:
            twos[topic].append(word[0])
## Calculate summary statistics for a unique or almost unique word being
# associated with its assigned top topic or a different topic
all_top_words = []
for topic in uniques.keys():
    for word in top_word[topic]:
        if word not in all_top_words:
            all_top_words.append(word)

maximum_non_assigned = 0
all_assigned = []
all_non_assigned = []
for idx,word in enumerate(all_top_words):
    all_of_them = model.get_term_topics(dictionary.token2id[str(word[0])])
    all_of_them = sorted(all_of_them, key=lambda x: x[1],reverse=True)
    if len(all_of_them) > 0:
        all_assigned.append(all_of_them[0][1])
    if len(all_of_them) > 1:
        all_non_assigned.append(all_of_them[1][1])
        if all_of_them[1][1] > maximum_non_assigned:
            maximum_non_assigned = all_of_them[1][1]
            print(word[0])
            print(all_of_them)

## Descriptive statistics for unique top words
print("mean (different topic): "+str(numpy.mean(all_non_assigned)))
print("max (different topic): "+str(maximum_non_assigned))
print("min (different topic): "+str(numpy.amin(all_non_assigned)))
print("standard deviation (different topic): "+str(numpy.std(all_non_assigned)))
print("***")
print("mean (assigned top topic): "+str(numpy.mean(all_assigned)))
print("max (assigned top topic): "+str(numpy.amax(all_assigned)))
print("min (assigned top topic): "+str(numpy.amin(all_assigned)))
print("standard deviation (assigned top topic): "+str(numpy.std(all_assigned)))


## Write unique terms for top topics to file. Change path if need be
fout = open("/Users/njjones14/Documents/Gene_Editing_Study/LDA_Models_Full_DataSet/LDA_True_25/top_uniques_{}".format(category),"a+")

for topic in top_word.keys():
    if topic in top_topics:
        print(topic,file=fout)
        # print len(uniques[topic])
        # print len(twos[topic])
        print("unique",file=fout)
        print(uniques[topic],file=fout)
        print("almost unique",file=fout)
        print(twos[topic],file=fout)

# ## Used in development of figures for publication. For developers' use only
# with open("LDA_{}_{}/top_twenty_50.txt".format(ENTIRE_CORPUS,num_topics),"w") as fout_new:
#     for topic in top_word.keys():
#         for idx,word_tuple in enumerate(top_word[topic]):
#             if idx == 20:
#                 break
#             print(word_tuple[0],file=fout_new)
#             print(word_tuple[1],file=fout_new)
