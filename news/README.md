This repository contains the libraries and scripts needed to run experiments
featured in a currently under-review study exploring methods for
classifying news/article links posted on Reddit, using only (non-content) user
behavior signals via graph embeddings and graph neural networks. As described
in the paper, the general experiment and analysis pipeline is as follows:

1. Identify a dataset of links l_1, l_2, ..., l_n and corresponding labels y_1, y_2, ..., y_n as targets for prediction.
2. Using a public data source like [pushshift.io](pushshift.io), for each link l_i, compute the time-ordered sequence of Reddit posts P_i = p_1, p_2, ... p_ti which contain the link as an embedded or text-referenced URL.
3. Let U = u_1, u_2, ..., u_m be the "complete" set of users, namely any user that authored or commented on a post containing any of the identified links. Construct an interaction-based derived social graph on U.
4. Generate graph embeddings V = v_1, v_2, ..., v_m, one for each user node.
5. Compute post-graphs for each post made of comment-reply edges. The author of each post is included as the root node.
6. Train a model predicting y_i from graph and embedding signals.

In the paper, we introduce a Recurrent Graph Neural Network (R-GNN) which, given a link l_i, trains a predictive layer for y_i by first encoding each post graph using V as node features, and then passing the GNN encodings to an RNN across posts.

The file which trains and evaluates the R-GNN is `main_comment_gcn.py`. Here is a description of the input files needed to run it:

* `seq_file`: An HDF file pandas DataFrame with two columns: `url` (the links l_1, l_2, ...) and `label` (the integer category for the link)
* `post_file`: An HDF file pandas DataFrame with two columns: `post_id` (the Reddit post ID identifying a Reddit post), and `user_list_and_edgelist` (a pairing of a list of Reddit user names and a dictionary mapping tuple edges to integer edge counts, each edge a pair of integer indices from the user name list)
* `balance_df`: (optional) An HDF file pandas DataFrame with the same columns as `seq_file`. This should contain a sub-sampled dataset to balance the category classes.
* `g_emb`: A keyed-vectors format file from the `gensim` package containing the graph embeddings V. This format is the return type of [KeyedVectors.load](https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.KeyedVectors.load).
