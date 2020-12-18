import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics.pairwise import pairwise_distances


def compute_similarity(df_embs):
    """
    Compute Similarity Between Items
    :param df_embs: Dataframe object with embeddings
    :return:
    """

    # Calculate Distance Matrix
    cosine_sim = 1 - pairwise_distances(df_embs, metric='cosine')
    # cosine_sim[:4, :4] # Visualize matrix
    return cosine_sim


def get_recommender(idx, df, df_embs, top_n=5):
    """
    Function that get fashion recommendations based on the cosine similarity score of fashion items
    :param df_embs: Dataframe object embeddings
    :param idx: image idx
    :param df: Dataframe obj
    :param top_n: Top n similar items
    :return: idx of recommended item, idx of similar items
    """
    # TODO: closely debug what the function does
    indices = pd.Series(range(len(df)), index=df.index)
    sim_idx = indices[idx]
    distance_matrix = compute_similarity(df_embs)
    sim_scores = list(enumerate(distance_matrix[sim_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1]
    idx_rec = [i[0] for i in sim_scores]
    idx_sim = [i[1] for i in sim_scores]

    return indices.iloc[idx_rec].index, idx_sim


def compute_similarity(df_embs):
    """
    Compute Similarity Between Items
    :param df_embs: Datafarame object with applied embeddings
    :return:
    """
    from sklearn.metrics.pairwise import pairwise_distances
    # Calculate Distance Matrix
    cosine_sim = 1 - pairwise_distances(df_embs, metric='cosine')
    return cosine_sim[:4, :4]