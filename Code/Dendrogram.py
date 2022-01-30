from gensim.models import LdaModel
from gensim.corpora import Dictionary

import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from defaults import model, num_topics

def main():
    # Load a pretrained model from disk.
    model = LdaModel.load(
        "/Users/njjones14/PycharmProjects/Big_Tech_Regulation/Models/" + str(num_topics) + "/LDA_" + str(num_topics))

    #Num_topics = 15
    # ["0/Telenor", "1/Healthcare", "2/Bartiromo and Trump", "3/Normative", "4/Big 4", "5/COIVD-19",
    #  "6/ Trump Administration", "7/FinTech",
    #  "8/Investment in Tech", "9/Antitrust and Regulation",
    #  "10/Corporate Growth", "11/Legal Power", "12/JUNK", "13/Share Price", "14/NPR Politics"]

    #Num_topics = 20
    # ["0/BT platforms", "1/Potential of Silicon Valley", "2/Antitrust and Big Tech", "3/COVID-19", "4/Trump",
    #  "5/Consumer Protection",
    #  "6/JUNK", "7/Inernational Secuirty",
    #  "8/Telenor", "9/Junk",
    #  "10/JUNK", "11/Investment in Tech", "12/BIG 4", "13/FinTech", "14/Tech Growth", "15/FinTech_2",
    #  "16/Privacy Law", "17/Normative", "18/JUNK", "19/International"]

    #NUM_TOPICS = 25
    # ["0/Silicon Valley", "1/Stock Price", "2/Consumer Protection Law", "3/Normative", "4/COVID-19 ",
    #  "5/International Big Tech", "6/JUNK", "7/FinTech", "8/News and Social Media", "9/JUNK",
    #  "10/JUNK", "11/Normative(JUNK)", "12/Corporate Growth", "13/BT Investigations", "14/China",
    #  "15/Expanding Business", "16/Antitrust", "17/DuckDuckGO", "18/Stock Price", "19/Children on Platforms",
    #  "20/Google's Behavioral \n Advertising", "21/Trump Administration", "22/AI Development",
    #  "23/Section 230", "24/JUNK"]
    # Load in the dictionary
    dictionary = ["0RealEstate","1ClimateChange","2New Tech in Business", "3Antitrust","4International", "5JUNK","6Antitrust","7CreditCards",
     "8ResearchFacialRegonition","9HealthCare","10Silicon Valley","11Content Moderation","12Tech Firms","13Positive Words",
     "14StockGrowth","15Telenor(JUNK)","16Growth","17Changing the world","18Trump in the News","19JUNK","20Facebook","21PlatfromContent",
     "22Stocks","23JUNK","24Tech in Cars(Uber / Tesla)","25JUNK","26Smatphones", "27ISPs","28International","29FinTech","30Politics","31COVID",
     "32Legislative Protection","33Earnings reports","34JUNK","35Workers","36Antitrust","37Googles International power","38Economy",
     "39NewTechPolicy","40CorporateFinance","41JUNK","42Growth_2","432020presidentialelection","44Privacy/personal Data",
     "45Investments","46CyberIntelligence","47JUNK","48JUNK","49JUNK"]

    # Get the distribution

    distribution = model.get_topics()

    # Remove junk indices
    junk_indexes = [0,1,4,5,7,10,15,19,23,24,25,26,27,28,29,33,34,35,36,41,43,47,48,49]
    remove_index = np.array(junk_indexes)
    # track array
    track_array = np.arange(50)

    # Remove the junk indices
    for i in range(len(remove_index)):
        current_index = remove_index[len(remove_index) - i - 1]
        distribution = np.delete(distribution, current_index, 0)
        track_array = np.delete(track_array, current_index, 0)

    # setting distance_threshold=0 ensures we compute the full tree.
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None, affinity="l1", linkage="complete")
    model.fit(distribution)
    plt.title('Hierarchical Clustering Dendrogram')

    # plot the top three levels of the dendrogram
    plot_dendrogram(model, truncate_mode='level', p=100, leaf_label_func=(lambda id: dictionary[track_array[id]]),
                    leaf_rotation=90)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.tight_layout()
    plt.show()


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        # counts[i] = dictionary[track_array[current_count]]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


if __name__ == "__main__":
    main()
