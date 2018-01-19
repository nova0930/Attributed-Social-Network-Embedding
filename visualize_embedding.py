import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE

def plot_embedding2D(node_pos, node_colors=None, di_graph=None):
    node_num, embedding_dim = node_pos.shape

    if(embedding_dim > 2):
        print("Embedding Dimension greater than 2, use tGNE to reduce it to 2")
        model = TSNE(n_components=2)
        node_pos = model.fit_transform(node_pos)

    if di_graph is None:
        plt.scatter(node_pos[:,0], node_pos[:,1], c=node_colors)
        labels = [i for i in range(node_num)]

        for label, x, y in zip(labels, node_pos[:, 0], node_pos[:, 1]):
            plt.annotate(
                label,
                xy=(x, y), xytext=(-2, 2),
                textcoords='offset points', ha='right', va='bottom')
    else:
        pos ={}
        for i in range(node_num):
            pos[i] = node_pos[i,:]

        if node_colors:
            nx.draw_networkx_edges(di_graph, pos, node_color= node_colors, width=0.1,node_size=100,arrows=False,
                                   alpha=0.8, font_size =5)
        else:
            # nx.draw_networkx(di_graph, pos, node_color=node_colors, width=0.1, node_size=300, arrows=False,
            #                        alpha=0.8, font_size=12)
            labels = {i:'G{0}'.format(i) for i in range(node_num)}

            nx.draw_networkx_nodes(di_graph, pos, node_size=100)
            nx.draw_networkx_edges(di_graph, pos)
            nx.draw_networkx_labels(di_graph, pos, labels)