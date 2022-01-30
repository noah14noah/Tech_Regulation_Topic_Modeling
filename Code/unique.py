import csv
import statistics
list_of_texts = []
with open("/Users/njjones14/PycharmProjects/Big_Tech_Regulation/list_of_texts.csv") as f:
    list_of_text_reader = csv.reader(f)
    for row in list_of_text_reader:
        list_of_texts.append(row)

with open("/Users/njjones14/PycharmProjects/Big_Tech_Regulation/list_of_texts_test.csv") as f:
    list_of_text_reader = csv.reader(f)
    for row in list_of_text_reader:
        list_of_texts.append(row)

words = []
counter = 0
lengths = []
for i in list_of_texts:
    lengths.append(len(i))
    for j in i:
        words.append(j)
        counter += 1


print(len(words))

unique = set(words)
print(counter, "counter")

print("# unique words = ", len(unique))

sd = statistics.stdev(lengths)
median = statistics.median(lengths)
print(sd, "sd")
print(median, "median")
