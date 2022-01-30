#Data visualization from Machine Learning plus (https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/)
#IMport
import pyLDAvis.gensim
from gensim.models import Phrases, CoherenceModel, LdaModel
import sys
# !{sys.executable} -m spacy download en
import re, numpy as np, pandas as pd
import seaborn as sns
import matplotlib.colors as mcolors
import csv
import plotly.express as px
import sys
# Gensim
import gensim, spacy, logging, warnings
from gensim.corpora import Dictionary
from defaults import num_topics

import matplotlib.pyplot as plt
warnings.filterwarnings("ignore",category=DeprecationWarning)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)


if __name__ == "__main__":

    # Load Necessary Files
    model = LdaModel.load(
        "/Users/njjones14/PycharmProjects/Big_Tech_Regulation/Models/" + str(num_topics) + "/LDA_" + str(num_topics))

    list_of_texts = []
    with open("/Users/njjones14/PycharmProjects/Big_Tech_Regulation/list_of_texts.csv") as f:
        list_of_text_reader = csv.reader(f)
        for row in list_of_text_reader:
            list_of_texts.append(row)
    print(list_of_texts)

    dictionary = Dictionary.load("/Users/njjones14/PycharmProjects/Big_Tech_Regulation/dictionary")
    corpus = [dictionary.doc2bow(doc) for doc in list_of_texts]

    with open("/Users/njjones14/PycharmProjects/Big_Tech_Regulation/corpus_cleaned.csv") as f:  # DATAFRAME
        corpus_cleaned = pd.read_csv(f)

    #TODO: What is the Dominant topic and its percentage contribution in each document
    def format_topics_sentences(ldamodel=None, corp=None, texts=None): #From ML+ Not my own code
        # Init output
        sent_topics_df = pd.DataFrame()

        # Get main topic in each document
        for i, row_list in enumerate(ldamodel[corp]):
            row = row_list[0] if ldamodel.per_word_topics else row_list
            # print(row)
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = ldamodel.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
                else:
                    break
        sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

        # Add original text to the end of the output
        contents = pd.Series(texts)
        sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
        return(sent_topics_df)

    df_topic_sents_keywords = format_topics_sentences(model, corpus, list_of_texts)

    # Format
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
    df_dominant_topic.to_csv("/Users/njjones14/PycharmProjects/Big_Tech_Regulation/Models/"+str(num_topics)+"/dominant_topics.csv")
    df_dominant_topic
    sys.exit(1)


    #TODO The most representative doc for each topic
    pd.options.display.max_colwidth = 100
    sent_topics_sorteddf_mallet = pd.DataFrame()
    sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

    for i, grp in sent_topics_outdf_grpd:
        sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet,
                                                 grp.sort_values(['Perc_Contribution'], ascending=False).head(1)],
                                                axis=0)

    # Reset Index
    sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

    # Format
    sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Representative Text"]

    # Show
    sent_topics_sorteddf_mallet.head(10)


    #TODO: Frequency Distribution of Word Counts in Documents

    doc_lens = [len(d) for d in df_dominant_topic.Text]

    # Plot
    plt.figure(figsize=(16,7), dpi=160)
    plt.hist(doc_lens, bins = 1000, color='navy')
    plt.text(750, 100, "Mean   : " + str(round(np.mean(doc_lens))))
    plt.text(750,  90, "Median : " + str(round(np.median(doc_lens))))
    plt.text(750,  80, "Stdev   : " + str(round(np.std(doc_lens))))
    plt.text(750,  70, "1%ile    : " + str(round(np.quantile(doc_lens, q=0.01))))
    plt.text(750,  60, "99%ile  : " + str(round(np.quantile(doc_lens, q=0.99))))

    plt.gca().set(xlim=(0, 1000), ylabel='Number of Documents', xlabel='Document Word Count')
    plt.tick_params(size=16)
    plt.xticks(np.linspace(0,1000,9))
    plt.title('Distribution of Document Word Counts', fontdict=dict(size=22))
    plt.show()

    # TODO:

    import seaborn as sns
    import matplotlib.colors as mcolors
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

    fig, axes = plt.subplots(2,2,figsize=(16,14), dpi=160, sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):
        df_dominant_topic_sub = df_dominant_topic.loc[df_dominant_topic.Dominant_Topic == i, :]
        doc_lens = [len(d) for d in df_dominant_topic_sub.Text]
        ax.hist(doc_lens, bins = 1000, color=cols[i])
        ax.tick_params(axis='y', labelcolor=cols[i], color=cols[i])
        sns.kdeplot(doc_lens, color="black", shade=False, ax=ax.twinx())
        ax.set(xlim=(0, 1000), xlabel='Document Word Count')
        ax.set_ylabel('Number of Documents', color=cols[i])
        ax.set_title('Topic: '+str(i), fontdict=dict(size=16, color=cols[i]))

    fig.tight_layout()
    fig.subplots_adjust(top=0.90)
    plt.xticks(np.linspace(0,1000,9))
    fig.suptitle('Distribution of Document Word Counts by Dominant Topic', fontsize=22)
    plt.show()

    #TODO: Word Clouds of Top N Keywords in Each Topic

    # 1. Wordcloud of Top N words in each topic
    from matplotlib import pyplot as plt
    from wordcloud import WordCloud, STOPWORDS
    import matplotlib.colors as mcolors

    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

    cloud = WordCloud(stopwords=stop_words,
                      background_color='white',
                      width=2500,
                      height=1800,
                      max_words=10,
                      colormap='tab10',
                      color_func=lambda *args, **kwargs: cols[i],
                      prefer_horizontal=1.0)

    topics = model.show_topics(formatted=False)

    fig, axes = plt.subplots(2, 2, figsize=(10,10), sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
        plt.gca().axis('off')


    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.show()

    # TODO: Word Counts of Topic Keyword
    from collections import Counter
    topics = model.show_topics(formatted=False)
    data_flat = [w for w_list in data_ready for w in w_list]
    counter = Counter(data_flat)

    out = []
    for i, topic in topics:
        for word, weight in topic:
            out.append([word, i , weight, counter[word]])

    df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])

    # Plot Word Count and Weights of Topic Keywords
    fig, axes = plt.subplots(2, 2, figsize=(16,10), sharey=True, dpi=160)
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
    for i, ax in enumerate(axes.flatten()):
        ax.bar(x='word', height="word_count", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.5, alpha=0.3, label='Word Count')
        ax_twin = ax.twinx()
        ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.2, label='Weights')
        ax.set_ylabel('Word Count', color=cols[i])
        ax_twin.set_ylim(0, 0.030); ax.set_ylim(0, 3500)
        ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=16)
        ax.tick_params(axis='y', left=False)
        ax.set_xticklabels(df.loc[df.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')
        ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')

    fig.tight_layout(w_pad=2)
    fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=22, y=1.05)
    plt.show()


    #Machine Learnign Plus Presenting results of LDA tutorial
    #https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/#6.-What-is-the-Dominant-topic-and-its-percentage-contribution-in-each-document
