#Generate data_for_R.csv, following Babak
import pandas as pd
import csv
from defaults import dictionary, model, corpus_cleaned, num_topics, list_of_texts

#parameters:
start_year = 2010
end_year = 2020


def transform_dates(cell):
    year = int(cell[0:4])
    month = int(cell[5:7])
    out_year = (year - start_year) + 1
    cell = (month,out_year)
    return cell

# Function to return most likely topic for each word in a document
def assign_topic_to_doc(doc):
    comment = doc["Content"].strip().split()
    bow = dictionary.doc2bow(comment)
    gamma, phis = model.inference([bow], collect_sstats=True)
    data_for_visualization = []
    for word_id, freq in bow:  # iterate over the word-topic assignments
        try:
            phi_values = [phis[i][word_id] for i in range(num_topics)]
        except KeyError:
            # Make sure the word either has a probability assigned to all
            # topics, or to no topics
            assert all(
                [word_id not in phis[i] for i in range(num_topics)]), "Word-topic probability assignment error"
            continue
        topic_asgmts = sorted(enumerate(phi_values), key=lambda x: x[1],
                          reverse=True)
        data_for_visualization.append(topic_asgmts[0][0])
    return data_for_visualization

if __name__ == "__main__":
    #Load files
    data_for_R = pd.DataFrame(columns=["number", "month", "year", "topic_assignments"])
    #create dataframe to make porportional-words-from-topics graphs

    for idx, doc in corpus_cleaned.iterrows():
        if int(doc["Date"][0:4]) >= 2010:
            date = transform_dates(doc["Date"])
            row = {"number": idx, "month": date[0], "year": date[1], "topic_assignments": assign_topic_to_doc(doc)}
            data_for_R = data_for_R.append(row, ignore_index=True)
        else: print(int(doc["Date"][0:4]))

    with open("/Users/njjones14/PycharmProjects/Big_Tech_Regulation/Models/" + str(num_topics) +"/data_for_R.csv", 'a+') as data:  # create the file
        fieldnames = ["number", "month", "year", "topic_assignments"]
        writer_R = csv.writer(data)  # initialize the CSV writer
        writer_R.writerow(fieldnames)
        for idx, comment in data_for_R.iterrows():
            writer_R.writerow([comment["number"], comment["month"], comment["year"], comment["topic_assignments"]])
    # data_for_R.to_csv("/Users/njjones14/PycharmProjects/Big_Tech_Regulation/Models/"+str(num_topics)+"/data_for_R")
