"""THE  Communication that a histogram-based the federated GBDT  (for example, the horizontal mode of FedTree) will be needed
to rebuild the  trained scikit-learn HistGradientBoostingClassifier exactly from the data spread over n clients.


Assumptions: each client takes part in the construction of every tree; for the root and for every split,
each client uploads one gradient histogram  (all bins of all of the  features, including the missing-value bin)  with
float32 sums of the gradients and Hessians, and the sibling histogram of a split is obtained by subtraction..
The number of synchronous exchanges is  bounded from below by one exchange per tree level (sum of the tree
depths); rebuilding a leaf-wise tree exactly needs one exchange per split.

Uses the  scikit-learn internals (_bin_mapper, _predictors), checked with scikit-learn 1.9.
"""


import numpy as np



def histogram_fl_cost(clf, n_clients=5, bytes_per_value=4):
    bins = clf._bin_mapper.n_bins_non_missing_ + 1             # bins per feature, plus the missing-value bin
    hist_bytes = int(bins.sum()) * 2 * bytes_per_value          # gradient and Hessian sum per bin
    depths, splits = [], []
    for it in clf._predictors:                                  # one tree per boosting iteration (binary task)
        nodes = it[0].nodes
        depths.append(int(nodes['depth'].max()))
        splits.append(int((~nodes['is_leaf'].astype(bool)).sum()))
    trees, n_split = len(depths), int(np.sum(splits))
    return dict(n_clients=n_clients, trees=trees, exchanges_min=int(np.sum(depths)), exchanges_leafwise=n_split,
                hist_bytes=hist_bytes, uplink_MB=n_clients * (n_split + trees) * hist_bytes / 1e6)

