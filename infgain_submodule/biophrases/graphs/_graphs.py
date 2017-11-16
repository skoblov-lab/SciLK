################
# Maxim Holmatov
################

import networkx as nx
from functools import partial
from multiprocessing import Pool

from nltk.corpus import stopwords
stopwords = stopwords.words('english')


__all__ = ["doc_to_graph", "corpus_to_graph", "centrality_measure",
           "best_bigram_from_graph", "best_bigram_from_doc",
           "bigrams_from_best", "bigrams_from_all"]


def doc_to_graph(document, g=None):
    """
    From parsed document create collocation graph

    :param document: parsed document
    :type document: sequence
    :param g: collocation graph to which new edges should be added
    :type g: networkx.classes.graph.Graph
    :returns: collocation graph for given document
    :rtype: networkx.classes.graph.Graph
    """
    if not g:
        g = nx.DiGraph()

    for first, second in zip(document, document[1:]):
        if g.has_edge(first, second):
            g[first][second]['weight'] += 1
        else:
            g.add_edge(first, second, weight=1)

    return g


def corpus_to_graph(corpus, g=None):
    """
    From parsed corpus create collocation graph

    :param corpus: parsed corpus
    :type corpus: sequence of sequences
    :param g: collocation graph to which new edges should be added
    :type g: networkx.classes.graph.Graph
    :returns: collocation graph for given corpus
    :rtype: networkx.classes.graph.Graph
    """
    g = nx.Graph()

    # for j in array:
    for j in corpus:
        for first, second in zip(j, j[1:]):
            if g.has_edge(first, second):
                g[first][second]['weight'] += 1
            else:
                g.add_edge(first, second, weight=1)

    return g


def centrality_measure(g, centrality):
    """
    Calculate centrality scores for vertices in graph

    :param g: collocation graph
    :type g: networkx.classes.graph.Graph
    :param centrality: type of centrality measure to use
    :type centrality: str
    :return: dictionary mapping graph vertices to their centrality measures
    :rtype: dict
    """
    cent = {'betweenness': partial(nx.betweenness_centrality,
                                   weight='weight'),
            'degree': nx.degree_centrality,
            'load': partial(nx.load_centrality, weight='weight'),
            'closeness': nx.closeness_centrality,
            'pagerank': partial(nx.pagerank, weight='weight')}

    return cent[centrality](g)


def best_bigram_from_graph(g, centrality):
    """
    From collocation graph return best bigram and it's score

    :param g: collocation graph
    :type g: networkx.classes.graph.Graph
    :param centrality: type of centrality measure to use
    :type centrality: str
    :return: tuple of bigram(tuple) and its centrality measure in this graph(float)
    :rtype: tuple
    """
    measure = centrality_measure(g, centrality)

    try:
        best = sorted([edge for edge in g.edges()
                      if edge[0] not in stopwords and
                      edge[1] not in stopwords and len(edge[0]) > 2 and
                      len(edge[1]) > 2],
                      key=lambda x: measure[x[0]] + measure[x[1]],
                      reverse=True)[0]
        return (best, measure[best[0]] + measure[best[1]])
    except IndexError:
        return (('inderr', 'inderr'), 0)


def best_bigram_from_doc(document, centrality):
    """
    From document return best bigram and it's score

    :param document: parsed document
    :type document: sequence
    :param centrality: type of centrality measure to use
    :type centrality: str
    :return: tuple of bigram(tuple) and its centrality measure in this graph(float)
    :rtype: tuple
    """
    return best_bigram_from_graph(doc_to_graph(document), centrality)


def bigrams_from_best(corpus, centrality, t=1):
    """
    From corpus return sorted list of best bigrams in each document
    and their centrality measures.

    :param corpus: parsed corpus
    :type corpus: sequence of sequences
    :param centrality: type of centrality measure to use
    :type centrality: str
    :param t: number of processes to dedicate
    :type t: int
    :return: list of tuples of bigrams(tuple) and their centrality measures
                                                        in this corpus
    :rtype: list
    """
    with Pool(processes=t) as p:
        best_bigrams = p.starmap(best_bigram_from_doc,
                                 [(i, centrality) for i in corpus],
                                 chunksize=len(corpus) // t)

    botb = {}

    for bigram in best_bigrams:
        if bigram[0] in botb:
            botb[bigram[0]] += bigram[1]
        else:
            botb[bigram[0]] = bigram[1]

    return [(key, value) for key, value in sorted(botb.items(),
                                                  key=lambda x: x[1],
                                                  reverse=True)]


def bigrams_from_all(corpus, centrality, n=1):
    """
    From corpus return sorted list of bigrams and their centrality measures.

    :param corpus: parsed corpus
    :type corpus: sequence of sequences
    :param centrality: type of centrality measure to use
    :type centrality: str
    :param n: number of processes to dedicate
    :type n: int
    :return: list of tuples of bigrams(tuple) and their centrality measures
                                                        in this corpus
    :rtype: list
    """
    g = corpus_to_graph(corpus)
    measure = centrality_measure(g, centrality)

    sort_edges = sorted(g.edges(), key=lambda x: measure[x[0]] + measure[x[1]],
                        reverse=True)

    return [(edge, measure[edge[0]] + measure[edge[1]]) for edge in sort_edges
            if edge[0] not in stopwords and
            edge[1] not in stopwords and len(edge[0]) > 2 and
            len(edge[1]) > 2]
