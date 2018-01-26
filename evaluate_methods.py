import pickle
import evaluation

def read_test_link(testlinkfile):
    X_test = []
    f = open(testlinkfile)
    line = f.readline()
    while line:
        line = line.strip().split(" ")
        X_test.append([int(line[0]), int(line[1]), int(line[2])])
        line = f.readline()
    f.close()
    print("test link number:", len(X_test))
    return X_test



# path = "./data/ecoli/"
# data_file = open("output/ecoli/processedData.pkl", "rb")
# Data = pickle.load(data_file)
# data_file.close()
# #
# print("Evaluating DeepWalk embeddings")
# embeddings_file = "output/ecoli/node2vec_embeddings/deep_emb.txt"
# emb = evaluation.load_embedding(embeddings_file, Data.id_N, combineAttribute=False, datafile=path+"data_standard.txt")
# X_test = read_test_link("./data/ecoli/edgelist_test.txt")
# roc = evaluation.evaluate_ROC(X_test, emb)
# print("Node2vec Method: Accuracy (ROC) in Test Data set", "{:.9f}".format(roc))
#
# embeddings_file = "output/ecoli/node2vec_embeddings/deep_emb.txt"
# emb = evaluation.load_embedding(embeddings_file, Data.id_N, combineAttribute=True, datafile=path+"data_standard.txt")
# X_test = read_test_link("./data/ecoli/edgelist_test.txt")
# roc = evaluation.evaluate_ROC(X_test, emb)
# print("Node2vec Method: Accuracy (ROC) in Test Data set", "{:.9f}".format(roc))


path = "./data/ecoli/"
data_file = open("output/ecoli/processedData.pkl", "rb")
Data = pickle.load(data_file)
data_file.close()
#

for test_size in [0.7, 0.6, 0.5, 0.4, 0.3, 0.2]:
    print("Evaluating DeepWalk embeddings")
    embeddings_file = "output/ecoli/deepWalk_embeddings/emb_"+str(test_size)+".txt"
    emb = evaluation.load_embedding(embeddings_file, Data.id_N, combineAttribute=False, datafile=path+"data_standard.txt")
    X_test = read_test_link("./data/ecoli/edgelist_test_"+str(test_size)+".txt")
    roc = evaluation.evaluate_ROC(X_test, emb)
    print("DeepWalk Method: Accuracy (ROC) in Test Data set", "{:.9f}".format(roc))
    #

    # #
    print("Evaluating LINE embeddings")
    embeddings_file = "output/ecoli/line_embeddings/emb_"+str(test_size)+".txt"
    emb = evaluation.load_embedding(embeddings_file, Data.id_N, combineAttribute=False, datafile=path+"data_standard.txt")
    roc = evaluation.evaluate_ROC(X_test, emb)
    print("LINE Method: Accuracy (ROC) in Test Data set", "{:.9f}".format(roc))
    # #


#
# print("Evaluating node2vec embeddings")
# embeddings_file = "output/ecoli/node2vec_embeddings_80.txt"
# emb = evaluation.load_embedding(embeddings_file, Data.id_N, combineAttribute=False, datafile=path+"data_standard.txt")
# X_test = read_test_link("./data/ecoli/edgelist_test.txt")
# roc = evaluation.evaluate_ROC(X_test, emb)
# print("node2vec Method: Accuracy (ROC) in Test Data set", "{:.9f}".format(roc))
#
# embeddings_file = "output/ecoli/node2vec_embeddings_80.txt"
# emb = evaluation.load_embedding(embeddings_file, Data.id_N, combineAttribute=True, datafile=path+"data_standard.txt")
# X_test = read_test_link("./data/ecoli/edgelist_test.txt")
# roc = evaluation.evaluate_ROC(X_test, emb)
# print("node2vec Method: Accuracy (ROC) in Test Data set", "{:.9f}".format(roc))


# for size in [30,40,50,60,70,80]:
#     print("Evaluating Line embeddings for training size: "+ str(size))
#     embeddings_file = "output/ecoli/node2vec_embeddings/emb_"+str(size)+".txt"
#     X_test = read_test_link("./data/ecoli/links_split/edgelist_test_"+str(size)+".txt")
#
#     emb = evaluation.load_embedding(embeddings_file, Data.id_N, combineAttribute=False, datafile=path+"data_standard.txt")
#     roc = evaluation.evaluate_ROC(X_test, emb)
#     print("DeepWalk: Accuracy (ROC) in Test Data set", "{:.9f}".format(roc))
#
# # for size in [30, 40, 50, 60, 70, 80]:
# print("Evaluating LINE embeddings for training size: " + str(size))
# embeddings_file = "output/ecoli/node2vec_embeddings/emb_" + str(size) + ".txt"
# X_test = read_test_link("./data/ecoli/links_split/edgelist_test_" + str(size) + ".txt")
#
# emb = evaluation.load_embedding(embeddings_file, Data.id_N, combineAttribute=False,
#                                 datafile=path + "data_standard.txt")
# roc = evaluation.evaluate_ROC(X_test, emb)
# print("LINE: Accuracy (ROC) in Test Data set", "{:.9f}".format(roc))