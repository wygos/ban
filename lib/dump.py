# reindexing nodes beacuse random forst and xgboost ues different numbering of nodes
def compute_index(tree):
    nodes_reindex = {}
    nodes = [(0,0)]
    next_node = 1
    while nodes:
        n, n_reindexed = nodes[0]
        nodes_reindex[n] = n_reindexed
        if tree.feature[n]>=0:
            nodes.append((tree.children_left[n], next_node))
            nodes.append((tree.children_right[n], next_node+1))
            next_node=next_node+2
        nodes.pop(0)
    return nodes_reindex


def dump_tree(tree, f, feature_names):
    nodes_reindex = compute_index(tree)

    nodes = [(0,0)]
    while nodes:
        n, indentation = nodes[0]
        nodes.pop(0)
        n_reindexed = nodes_reindex[n]
        if tree.feature[n]>=0:
            left = tree.children_left[n]
            right = tree.children_right[n]
            f.write('\t'*indentation+f'{n_reindexed}:[{feature_names[tree.feature[n]]}<{tree.threshold[n]}] \
yes={nodes_reindex[left]},no={nodes_reindex[right]},missing={nodes_reindex[left]},gain=0.0')
            nodes.insert(0, (right, indentation+1))
            nodes.insert(0, (left, indentation+1))
        else:
            f.write('\t'*indentation+f'{n_reindexed}:leaf={tree.value[n][0][0]}')
        # shap uses weighted_n_node_samples instead of n_node_samples
        f.write(f',cover={tree.weighted_n_node_samples[n]}\n')

def dump_trees(model, FNAME, feature_names, model_type):
    if model_type == 'DT':
        with open(FNAME, "w") as f:
            for i, est in enumerate(model.estimators_):
                f.write(f'booster[{i}]:\n')
                dump_tree(est.tree_, f, feature_names)
    else:
        assert model_type=='GBDT'
        model.dump_model(FNAME, with_stats=True)


