This repository contains the libraries and scripts needed to run experiments
featured in a currently under-review study exploring methods for
classifying news/article links posted on Reddit, using only (non-content) user
behavior signals via graph embeddings and graph neural networks. As described
in the paper, our experiment and analysis pipeline is as follows:

1. Identify a dataset of links $$\{l_1, l_2, \ldots, l_n\}$$ and corresponding labels $$\{y_1, y_2, \ldots, y_n\}$$ as targets for prediction.
2. Using a public data source like [pushshift.io](pushshift.io), for each link $$l_i$$, compute the time-ordered sequence of Reddit posts $$\mathbf{p}_i = \{p_1, p_2, \ldots, p_{t_i}\}$$ which contain the link as an embedded or text-referenced URL.
3. Let $$\mathbf{U} = \{u_1, u_2, \ldots, u_m\}$$ be the "complete" set of users, namely any user that authored or commented on a post within any element of $$\mathbf{P} = \{\mathbf{p}_1, \mathbf{p}_2, \ldots, \mathbf{p}_n\}$$. Construct an interaction-based derived social graph $$\mathbf{G}$$ on $$\mathbf{U}$$. $$\mathbf{G}$$ is comprised of the node set $$\mathbf{U}$$ and an undirected edge set $$\mathbf{E}$$ where $$\{u_i, u_j\}\in \mathbf{E}$$ iff $$u_i$$ and $$u_j$$ are connected in the graph.
4. Generate graph embeddings $$\mathbf{V} = \{v_1, v_2, \ldots, v_m\}$$, one for each user node in $$\mathbf{G}$$.
5. Compute post-graphs $$\mathbf{H} = \{\mathbf{h}_1, \mathbf{h}_2, \ldots, \mathbf{h}_n\}$$ where $$\mathbf{h}_i = (\mathbf{u}_i, \mathbf{e}_i)$$, $$\mathbf{u}_i$$ is the set of users who commented on post $$i$$, and $$\mathbf{e}_i$$ is the set of comment reply-edges. The author of post $$i$$ is also included as a node in $$\mathbf{h}_i$$.
6. Train a model predicting $$y_i$$ from any signals of $$\mathbf{G}$$, $$\mathbf{V}$$, and $$\mathbf{H}$$.

In the forthcoming paper, we introduce a Recurrent Graph Neural Network (R-GNN) which, given a link $$l_i$$, trains a predictive layer for $$y_i$$ by first encoding each post graph $$\mathbf{h}$$ corresponding to a post in $$\mathbf{p}_i$$, using $$\mathbf{V}$$ as node features, and then passing the GNN encodings to an RNN across $$\mathbf{p}_i$$.

The file which trains and evaluates the R-GNN is `main_comment_gcn.py`. Here is a description of the input files needed to run it:

* `seq_file`: An HDF file pandas DataFrame with two columns: `url` (the links $$l_1, l_2, \ldots$$) and `label` (the integer category for the link)
* `post_file`: An HDF file pandas DataFrame with two columns: `post_id` (the Reddit post ID identifying a Reddit post), and `user_list_and_edgelist` (a pairing of a list of Reddit user names and a dictionary mapping tuple edges to integer edge counts, each edge a pair of integer indices from the user name list)
* `balance_df`: (optional) An HDF file pandas DataFrame with the same columns as `seq_file`. This should contain a sub-sampled dataset to balance the category classes.
* `g_emb`: A keyed-vectors format file from the `gensim` package containing the graph embeddings $$\mathbf{V}$$. This format is the return type of [KeyedVectors.load](https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.KeyedVectors.load).
