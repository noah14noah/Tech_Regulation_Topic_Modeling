#Summed Topic Contributions, for plot_monthly_topic_contributions.py
#Imports
from gensim.models import Phrases, CoherenceModel, LdaModel
from gensim.corpora import Dictionary
import pandas as pd
import csv
import sys

def transform_dates(cell):
    s_year = 1980
    e_year = 2020 + 1
    year = int(cell[0:4])
    print(year)
    month = int(cell[5:7])
    print(month)
    cell = ((year - s_year) * 12) + (month - 1)
    print(cell)
    return cell

if __name__ == "__main__":
    # Parameters
    num_topics = 30
    s_year = 1980
    e_year = 2020 + 1
    months_range = (len(range(s_year, e_year)) * 12)  # Todo: confirm dates after having picked data

    # Load Necessary Files
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


    def format_topics_contributions(ldamodel=None, corpus=None):
        get_doc_topics = [ldamodel.get_document_topics(i) for i in corpus]
        # array of topic distributions series of (int, float), length = corpus_cleaned
        # print(len(get_doc_topics))
        # print(get_doc_topics[0])
        # print(model[corpus[0]])
        doc_topic_contribution = pd.DataFrame({"topic_contributions to document": get_doc_topics})
        # print(len(doc_topic_contribution))
        # TODO: ask babak if this is a safe what to do it, considering that I am just lining them up
        # TODO: also assumes that document topics are in assending order (0-num_topics)
        # end goal is to have (Doc_Number,[(int,float),(int,float),(int,float)...])
        corpus_cleaned['Topic_Contributions'] = doc_topic_contribution
        # print("TESTING:")
        # print("are these equal?")
        df_w_contributions = corpus_cleaned
        return df_w_contributions


    df_w_topics = format_topics_contributions(model, corpus)
    # Should
    print(df_w_topics)

    print(df_w_topics.columns)
    #transform dates

    print(df_w_topics["Date"][2240:2242])


    df_w_topics['Date'] = df_w_topics['Date'].map(transform_dates)
    print(df_w_topics["Topic_Contributions"])
    #Here, I should have a pandas dataframe that is of length of the corpus, with all the necessary columns, including the final colummn with all the topic distribu

    def summed_topics_contributions(df):
        if not type(df) == pd.DataFrame:
            raise Exception("input df is not DataFrame")
        else:
            # initalize ouput dataframe
            topic_list = [i for i in range(num_topics)]
            months = [i for i in range(months_range)]
            topic_over_months = pd.DataFrame(0, columns=months, index=topic_list)
            print(topic_over_months)
            for idx, row in df.iterrows():
                for p in row["Topic_Contributions"]:
                    # print(p[0])
                    # print(p[1])
                    # print(p[0], row["Date"], p[1])
                    # print(idx)
                    # print(row)
                    # print(topic_over_months.loc[p[0], row["Date"]])
                    topic_over_months.loc[p[0], row["Date"]] += p[1]
            print(topic_over_months)
        return topic_over_months #TODO; These should be average, to make more interesting
    fout = summed_topics_contributions(df_w_topics)
    print(fout)

    fout.to_csv("/Users/njjones14/PycharmProjects/Big_Tech_Regulation/Models/" + str(num_topics) + "/monthly_topic_contribution")

    # Should have now have summed-monthly contributions for each topic