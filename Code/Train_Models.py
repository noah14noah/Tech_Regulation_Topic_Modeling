#Train Models
from gensim.models import LdaModel, CoherenceModel
from defaults import dictionary, corpus, corpus_test
import numpy as np

if __name__ == "__main__":
    print('Number of unique tokens: {}'.format(len(dictionary)))
    print('Number of documents: {}'.format(len(corpus)))
    print('Number of documents in test: {}'.format(len(corpus_test)))

    #Training parameters
    chunksize = 1000
    passes = 20
    iterations = 1000
    eval_every = 1

    # id2word = dictionary.id2token
    id2word = dictionary
    print(id2word)

    num_topics_list = [25,75,100] # Need to create "Models" and "num_topic" folder in directory
    for num_topics in num_topics_list:
        model = LdaModel(
            corpus=corpus,
            id2word=id2word,
            chunksize=chunksize,
            alpha='auto',
            eta='auto',
            iterations=iterations,
            num_topics=num_topics,
            passes=passes,
            eval_every=eval_every,
            minimum_probability=0.01
        )
        # print topics
        print(model.print_topics(num_words=40))
        # Save Model
        model.save("/Users/njjones14/PycharmProjects/Big_Tech_Regulation/Models/" + str(num_topics) + "/LDA_" + str(num_topics))
        model = LdaModel.load("/Users/njjones14/PycharmProjects/Big_Tech_Regulation/Models/"+str(num_topics)+"/LDA_"+str(num_topics))
        with open("/Users/njjones14/PycharmProjects/Big_Tech_Regulation/Models/"+str(num_topics)+"/topics_"+str(num_topics), "w") as f:
            for idx, topic in model.print_topics(num_topics=num_topics, num_words=40):
                # model.print_topic
                # print("I think it is an number of topic but it is actually: " + str(idx))
                # print("I think this is the topic, actually: " + topic)
                # print(topic[0])
                # print(type(topic[1]))
                print("Topic: {} \n Probability and Words: {}".format(idx, topic), file=f)
        # Compute Perplexity
        print('\nPerplexity: ', model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.
        # Compute Coherence Score
        cm = CoherenceModel(model=model, corpus=corpus, coherence='u_mass')
        coherence = cm.get_coherence()  # get coherence value
        print('\nCoherence Score: ', coherence)
        print("Model with " + str(num_topics) + " topics is finished")

        with open("/Users/njjones14/PycharmProjects/Big_Tech_Regulation/Models/"+str(num_topics) +"/LDA_"+str(num_topics)+"_preformance", "w") as performance:
            # train perplexity scores
            train_perplexity = model.bound(corpus, subsample_ratio = .99)
            eval_perplexity = model.bound(corpus_test, subsample_ratio = .01)
            # calculate per-word perplexity for training and evaluation sets
            train_word_count = 27566286 # manually generated
            eval_word_count = 230798 # manually geereated
            train_per_word_perplex = np.exp2(-train_perplexity / train_word_count)
            eval_per_word_perplex = np.exp2(-eval_perplexity / eval_word_count)
            performance.write("Performance")
            performance.write("\n")
            performance.write("Coherence")
            performance.write("\n")
            performance.write(str(coherence))
            performance.write("\n")
            performance.write("Perplexity_train")
            performance.write("\n")
            performance.write(str(train_per_word_perplex))
            performance.write("\n")
            performance.write("Perplexity_eval")
            performance.write("\n")
            performance.write(str(eval_per_word_perplex))

        # ## find top words associated with EVERY topic and write them to file
        # # Using code from babak LDA_analysis.py
        # top_words_all = {key: [] for key in range(num_topics)}
        # with open("/Users/njjones14/PycharmProjects/Big_Tech_Regulation/Models/" + str(
        #         num_topics) + "/top_words_all_" + str(num_topics), 'a+') as f:
        #     # create a file for storing the high-probability words
        #     for topic in top_words_all.keys():
        #         print(topic, file=f)
        #         output = model.show_topic(topic, topn=80)
        #         print(output, file=f)
        #         top_words_all[topic] = model.show_topic(topic, topn=80)  # Babak uses topn=80 in defaults.py
        # print("Model with " + str(num_topics) + "if completed")