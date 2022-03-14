#include <Python.h>
#include "../shap/shap/cext/tree_shap.h"

#include <cstddef>
#include <queue>
#include <cstring>
    
TreeEnsemble get_tree_ensamble(Trees const & Ts) {
    unsigned max_nodes = 0;
    unsigned max_depth = 0;
    unsigned tree_limit = Ts.size();
    auto base_offset = new tfloat[Ts.size()];
    std::memset (base_offset, 0., Ts.size()* sizeof(tfloat));
    for(auto const & t : Ts) {
        max_nodes = std::max(max_nodes, (unsigned)t->size());
        max_depth = std::max(max_depth, (unsigned)t->height());
    }
    auto n = Ts.size()* max_nodes;
    //TODO memory leak 
    auto children_left = new int[n];
    auto children_right = new int [n];
    auto children_default = new int[n];
    auto features = new int[n];
    auto thresholds = new tfloat[n];
    auto values = new tfloat[n];
    auto node_sample_weights = new tfloat[n];

    for(auto i =0; i < n; ++i) {
       children_left[i] = -1;
       children_right[i] = -1;
       children_default[i] = -1;
       features[i] = -1;
    }
    std::memset (thresholds, 0., n* sizeof(tfloat));
    std::memset (values, 0., n* sizeof(tfloat));
    std::memset (node_sample_weights, 0., n* sizeof(tfloat));
    
    
    for(std::size_t tree_idx = 0; tree_idx < Ts.size(); ++tree_idx) {
        auto offset = tree_idx * max_nodes;
        auto node_idx = 0;
        std::queue<std::tuple<decision_tree *, int, bool, bool>> nodes;
        nodes.push(std::make_tuple(Ts[tree_idx], -1, false, false));
        while(!nodes.empty()) {
            decision_tree * node;
            int par;
            bool is_left, is_default;
            std::tie(node, par, is_left, is_default) =  nodes.front();
            nodes.pop();
            auto idx = offset + node_idx;
            //TODO different definbition used here
            node_sample_weights[idx] = node->prob;

            if (par != -1) {
                if (is_left) {
                    children_left[par] = node_idx;
                } else {
                    children_right[par] = node_idx;
                }
                if (is_default) {
                  children_default[par] = node_idx;
                }
            } 
            if(node->is_leaf()) {
                values[idx] = node->t;
            } else {
                features[idx] = node->feature;
                thresholds[idx] = node->t;
                nodes.push(std::make_tuple(node->left, idx, true, node->missing_go_left));
                nodes.push(std::make_tuple(node->right, idx, false, !node->missing_go_left));
            }
            ++node_idx;
        }

    }
    auto num_outputs = 1;
    return TreeEnsemble(children_left, children_right, children_default, features,
                 thresholds, values, node_sample_weights,
                 max_depth, tree_limit, base_offset,
                 max_nodes, num_outputs);
}

std::vector<res_row> compute_original(Trees const & Ts, const Data &data_mim) {
    auto row_size = data_mim[0].size();
    std::vector<res_row> res(data_mim.size(), res_row(row_size));
    auto model_transform = 0;
    auto transform_f = get_transform(model_transform);

    auto trees = get_tree_ensamble(Ts);

    std::vector<tfloat> X_vec(data_mim.size() * row_size, 0);
    bool *  X_missing = new bool[data_mim.size() * row_size]; 
    std::vector<tfloat> y_vec(0);
    std::vector<tfloat> R_vec(0); //TODO what is it?
    std::vector<bool> R_missing_vec(0); //TODO what is it?
    
    
    for(std::size_t row_idx = 0; row_idx < data_mim.size(); ++row_idx) {
        for(std::size_t idx = 0; idx < row_size; ++idx) {
            X_vec[row_idx * row_size + idx] = data_mim[row_idx][idx];
            X_missing[row_idx * row_size + idx] = std::isnan(data_mim[row_idx][idx]);
        }
    }

                       
    ExplanationDataset data(X_vec.data(), X_missing, nullptr, nullptr, nullptr, data_mim.size(),row_size, 0);
    //ExplanationDataset data(X_vec.data(), X_missing_vec.data(), y_vec.data(), R_vec.data(), R_missing_vec.data(), row_size,
    //                   data_mim.size(), data_mim.size()) :
    //#TODO missing  PyArg_ParseTuple ???
    //PyObject *out_contribs_obj;
    //PyArrayObject *out_contribs_array = (PyArrayObject*)PyArray_FROM_OTF(out_contribs_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY);
    //tfloat *out_contribs = (tfloat*)PyArray_DATA(out_contribs_array);
    std::vector<tfloat> out_contribs_vec(data_mim.size()*(row_size+1));
    dense_tree_path_dependent(trees, data, out_contribs_vec.data(), transform_f);
    for(std::size_t row_idx = 0; row_idx < res.size(); ++row_idx) {
        for(std::size_t idx = 0; idx < row_size; ++idx) {
            res[row_idx][idx] = out_contribs_vec[row_idx * (row_size+1)+ idx];
        }
    }

    delete[] X_missing;

    return res;
}
