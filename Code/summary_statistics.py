import csv
import statistics
import pandas as pd

# list_of_texts = []
# with open("/Users/njjones14/PycharmProjects/Big_Tech_Regulation/list_of_texts.csv") as f:
#     list_of_text_reader = csv.reader(f)
#     for row in list_of_text_reader:
#         list_of_texts.append(row)
#
# with open("/Users/njjones14/PycharmProjects/Big_Tech_Regulation/list_of_texts_test.csv") as f:
#     list_of_text_reader = csv.reader(f)
#     for row in list_of_text_reader:
#         list_of_texts.append(row)
#
# words = []
# counter = 0
# lengths = []
# for i in list_of_texts:
#     lengths.append(len(i))
#     for j in i:
#         words.append(j)
#         counter += 1
#
#
# unique = set(words)
# print(counter, "counter")
#
# print("# unique words = ", len(unique))
#
# sd = statistics.stdev(lengths)
# median = statistics.median(lengths)
# print(sd, "sd")
# print(median, "median")

corpus_clean = pd.read_csv("/Users/njjones14/PycharmProjects/Big_Tech_Regulation/corpus_cleaned.csv")
print(corpus_clean)
def transform_dates(cell):
    year = cell[0:4]
    return year

year_dict = {}

for row in corpus_clean["Date"]:
    print(row)
    year = int(transform_dates(row))
    if str(year) in year_dict.keys():
        year_dict[str(year)].append(year)
    else:
        year_dict[str(year)] = [year]

for i, j in year_dict.items():
    print("year", i)
    print("num_comments", len(j))
