#inital Import and Clean of Json files files form Lexis Nexus
import re
import unicodedata
import contractions
import os
from nltk.corpus import stopwords
import csv
stopwords = (stopwords.words('english'))
import unidecode
from bs4 import BeautifulSoup
from spacy.lemmatizer import Lemmatizer
import spacy
import ijson
import codecs
import pandas as pd
from spacy.lookups import Lookups
from gensim.models import Phrases
from gensim.corpora import Dictionary
from langdetect import detect
import numpy as np
from sklearn.model_selection import train_test_split
import statistics
#Manual stopwords dictionary:
manual_stopwords = {"organizationalfacebookpage":"organizational facebook page","thatfacebook" : "that facebook",
                    "fromfacebookemployee" : "from facebook employee", "fromfacebookemployees":"from facebook employees",
                    "facebookaccount" : "facebook account"}

#TODO: finisd industry and subject tags
industries = pd.read_excel("/Users/njjones14/PycharmProjects/Big_Tech_Regulation/English_Industry_Taxonomy.xlsx", names = ["Industry"])
industry_list = []
for idx, row in industries.iterrows():
    industry_list.append(row["Industry"])

subjects = pd.read_excel("/Users/njjones14/PycharmProjects/Big_Tech_Regulation/English_Subject_Taxonomy.xlsx", names = ["Subject"])
subject_list=[]
for idx, row in subjects.iterrows():
    subject_list.append(row["Subject"])

class Row_object:
    def __init__(self, ID, Jurisdiction, Location, ContentType, Byline, WordLength, Date, Title, Content, SourceName):
        # , industry_dict={}, subject_dict={}
        self.ID = ID
        self.Jurisdiction = Jurisdiction
        self.Location = Location
        self.ContentType = ContentType
        self.Byline = Byline
        self.WordLength = WordLength
        self.Date = Date
        self.Title = Title
        self.Content = Content
        self.SourceName = SourceName
        # self.industry_dict = industry_dict
        # self.subject_dict = subject_dict


# Pre-Tokenization functions
def Soup_to_bodyText(text):
    soup1 = BeautifulSoup(text, "html.parser")
    doc_content = str(soup1.find("nitf:body.content"))
    #remove urls
    soup2 = BeautifulSoup(doc_content, "html.parser")
    for s in soup2.select("url"):
        s.extract()
    #remove
    return str(soup2)

def remove_html_tags(text):
    """Remove html tags from a string"""
    clean = re.compile('<.*?>')
    return re.sub(clean, ' ', text)

def remove_accented_chars(text):
    """remove accented characters from text, e.g. café"""
    text = unidecode.unidecode(text)
    return text

def replace_contractions(text):
    """Replace contractions in string of text"""
    return contractions.fix(text)

def remove_new_line_characters(text):
    """Replace new line characters"""
    return re.sub('\s+', ' ', text)

def remove_bracketed_text(text):
    doc_content = re.sub("[\(\[].*?[\)\]]", " ", text)
    return doc_content

# def remove_long_short_words(text): #This should be done after stop words and punctuation
#     list_of_removed = []
#     for word in text.split():
#         if len(word) > 20 or len(word) < 2:
#             list_of_removed.append(word)
#             text = text.replace(word, "")
#     print(list_of_removed)
#     return text
# doc_content = remove_long_short_words(doc_content)

# TODO: POST-Tokenization _____________________

lookups = Lookups()
lookups.add_table("lemma_rules", {"noun": [["s", ""]]})
lemmatizer = Lemmatizer(lookups)

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords:
            new_words.append(word.strip())
    return new_words

def remove_small_words(words):
    for word in words:
        if len(word.strip()) <= 2:
            words.remove(word)
    return words

def manual_fixes(words):
    out_list = []
    for word in words:
        if word.lower() in manual_stopwords.keys():
            out_list.append(manual_stopwords[word])
        else:
            out_list.append(word)
    return out_list

# def Soup_to_industry_tags(text):
#     soup1 = BeautifulSoup(text, "html.parser")
#     doc_content = str(soup1.find("metadata"))
#     soup2 = BeautifulSoup(doc_content, "html.parser")
#     industry_list_list= []
#     industry_dict = {}
#     for s in soup2.find_all('classification'):
#         print(s)
#         soup3 = BeautifulSoup(str(s), "html.parser")
#         for t in soup3.find_all('classificationitem'):
#             if t.classname.get_text() in industry_list:
#                 score = t["score"]
#                 tag = t.classname.get_text()
#                 industry_dict[tag] = int(score)
#     return industry_dict
#
# def Soup_to_subject_tags(text):
#     soup1 = BeautifulSoup(text, "html.parser")
#     doc_content = str(soup1.find("metadata"))
#     soup2 = BeautifulSoup(doc_content, "html.parser")
#     subjects_list = []
#     for s in soup2.find_all('classification'):
#         if "subject" in str(s):
#             subjects_list.append(s)
#     for i in subjects_list:
#         if "licensor" not in i:
#             correct = str(i)
#     soup3 = BeautifulSoup(correct, "html.parser")
#     subject_dict = {}
#     for s in soup3.find_all('classificationitem'):
#         score = s["score"]
#         tag = s.classname.get_text()
#         subject_dict[tag] = int(score)
#     return subject_dict

def LDA_clean(input_content):
    soup_content = Soup_to_bodyText(input_content)
    html_content = remove_html_tags(soup_content)
    accented_content = remove_accented_chars(html_content)
    contract_content = replace_contractions(accented_content)
    new_line_content = remove_new_line_characters(contract_content)
    token_input = remove_bracketed_text(new_line_content)
    ##TOKENIZE
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(token_input)
    doc = [token.lemma_ if token.lemma_ != '-PRON-' else token.text for token in doc] #TODO: remove words at below certian frequency? \
    non_ascii = remove_non_ascii(doc)
    lower_content = to_lowercase(non_ascii)
    sans_punctuation = remove_punctuation(lower_content)
    no_small_words = remove_small_words(sans_punctuation)
    sans_stops = remove_stopwords(no_small_words)
    fixed_manually = manual_fixes(sans_stops)
    return " ".join(fixed_manually)

directory = '/Users/njjones14/PycharmProjects/Big_Tech_Regulation/Big_Tech_Regulation_data/'

def parse_single_file(filename):
    with open(directory + filename) as f:
        content = ijson.items(f, "value")
        row_list = []
        for o in content:
            print(filename)
            for i in range(0, len(o)):
                print(type(o[i]["Document"]))
                if o[i]["Document"] is not None: #Drops rows where there is no content
                    prop_id = np.random.random_integers(low=10000000, high=99999999)
                    row_list.append(Row_object(prop_id, o[i]["Jurisdiction"], o[i]["Location"], o[i]["ContentType"],
                                               o[i]["Byline"], o[i]["WordLength"], o[i]["Date"], o[i]["Title"],
                                               LDA_clean(o[i]["Document"]["Content"]), o[i]["Source"]["Name"]))
    return row_list

# ,
#                                                Soup_to_industry_tags(o[i]["Document"]["Content"]),
#                                                Soup_to_subject_tags(o[i]["Document"]["Content"])

list_of_files = []
for file in sorted(os.listdir(directory)):
    if not file.startswith("."):
        list_of_files.append(file)


if __name__ == "__main__":
    for filename in list_of_files:
        list_of_row_objects = parse_single_file(filename)
        with open("/Users/njjones14/PycharmProjects/Big_Tech_Regulation/cleaned_csv_files/" + filename[0:17] + ".csv", "w") as csvfile:
            row_writer = csv.writer(csvfile)
            for i in list_of_row_objects:
                row_writer.writerow(
                    [i.ID, i.Title, i.Byline, i.Date, i.Jurisdiction, i.Location, i.ContentType, i.WordLength, i.Content,
                     i.SourceName])

    # , i.industry_dict, i.subject_dict
    csv_list = []
    for csv_file in sorted(os.listdir("/Users/njjones14/PycharmProjects/Big_Tech_Regulation/cleaned_csv_files/")):
        if not csv_file.startswith("."):
            df = pd.read_csv("/Users/njjones14/PycharmProjects/Big_Tech_Regulation/cleaned_csv_files/" + str(csv_file), header=None)
            csv_list.append(df)
    result = pd.concat(csv_list)

    df_columns = ["ID","Title", "Byline", "Date", "Jurisdiction", "Location", "ContentType", "WordLength", "Content",
                  "SourceName"]

    result.columns = df_columns
    wo_duplicates = result.drop_duplicates(subset="Title") #remove duplicates before writing to file
    wo_duplicates = wo_duplicates.dropna(axis=0, subset=["Content"])
    wo_duplicates.to_csv("/Users/njjones14/PycharmProjects/Big_Tech_Regulation/corpus_all_years.csv")

    plus2010 = pd.DataFrame(columns=["ID","Title", "Byline", "Date", "Jurisdiction", "Location", "ContentType", "WordLength", "Content",
                  "SourceName"])
    for idx, row in wo_duplicates.iterrows():
        if int(row["Date"][0:4]) >= 2010:
            row = {"ID": row["ID"], "Title": row["Title"], "Byline": row["Byline"], "Date": int(row["Date"][0:4]),
                   "Jurisdiction": row["Jurisdiction"], "Location": row["Location"], "ContentType": row["ContentType"],
                                   "WordLength": row["WordLength"], "Content": row["Content"], "SourceName": row["SourceName"]}
            plus2010 = plus2010.append(row, ignore_index=True)


    #set training and testing sets, saving to file
    train, test = train_test_split(plus2010, test_size=0.01)

    train.to_csv("/Users/njjones14/PycharmProjects/Big_Tech_Regulation/training_set.csv")
    test.to_csv("/Users/njjones14/PycharmProjects/Big_Tech_Regulation/testing_set.csv")

    list_of_texts = []
    for idx, row in train.iterrows():
        if detect(row["Content"]) == "en":
            list_of_texts.append(list(row["Content"].split(" ")))

    list_of_texts_test = []
    for idx, row in test.iterrows():
        if detect(row["Content"]) == "en":
            list_of_texts_test.append(list(row["Content"].split(" ")))

    #bigrams
    bigram = Phrases(list_of_texts, min_count=20, threshold=100)

    for idx in range(len(list_of_texts)):
        for token in bigram[list_of_texts[idx]]:
            if '_' in token:
                # Token is a bigram, add to document.
                list_of_texts[idx].append(token)


    # Create Dictionaries
    dictionary = Dictionary(list_of_texts)
    dictionary_test = Dictionary(list_of_texts_test)
    # save dictionaries to file
    dictionary.save("/Users/njjones14/PycharmProjects/Big_Tech_Regulation/dictionary")
    dictionary_test.save("/Users/njjones14/PycharmProjects/Big_Tech_Regulation/dictionary_test")

    # Read list_of_texts (with bigrams) to file
    with open("/Users/njjones14/PycharmProjects/Big_Tech_Regulation/list_of_texts.csv", "w") as file:
        writer = csv.writer(file)
        writer.writerows(list_of_texts)
    with open("/Users/njjones14/PycharmProjects/Big_Tech_Regulation/list_of_texts_test.csv", "w") as file:
        writer = csv.writer(file)
        writer.writerows(list_of_texts_test)

    ################## Summary Statistics ####################
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

    outF = open("/Users/njjones14/PycharmProjects/Big_Tech_Regulation/summary_stats.txt", "w")

    all_list_of_texts = list_of_texts + list_of_texts_test
    words = []
    counter = 0
    lengths = []

    for i in all_list_of_texts:
        lengths.append(len(i))
        for j in i:
            words.append(j)
            counter += 1

    unique = set(words)
    print(counter, "counter")
    outF.write("# of articles: ")
    outF.write(str(len(lengths)))
    outF.write("\n")
    outF.write("# of words: ")
    outF.write(str(counter))
    outF.write("\n")
    outF.write("# unique words = " )
    outF.write(str(len(unique)))
    outF.write("\n")
    print("# unique words = ", str(len(unique)))

    sd = statistics.stdev(lengths)
    median = statistics.median(lengths)
    print(sd, "sd")
    outF.write("sd: ")
    outF.write(str(sd))
    outF.write("\n")
    print(median, "median")
    outF.write("median: ")
    outF.write(str(median))
    outF.write("\n")

    with open("/Users/njjones14/PycharmProjects/Big_Tech_Regulation/corpus_all_years.csv") as f:  # DATAFRAME
        corpus_cleaned = pd.read_csv(f)

    def transform_dates(cell):
        year = cell[0:4]
        return year

    year_dict = {}

    for row in corpus_cleaned["Date"]:
        print(row)
        year = int(transform_dates(row))
        if str(year) in year_dict.keys():
            year_dict[str(year)].append(year)
        else:
            year_dict[str(year)] = [year]

    for i, j in year_dict.items():
        print("year", i)
        outF.write("year: ")
        outF.write(str(i))
        outF.write("\n")
        print("num_comments", len(j))
        outF.write("num_comments: ")
        outF.write(str(len(j)))
        outF.write("\n")


