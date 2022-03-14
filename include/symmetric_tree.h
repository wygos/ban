#include <vector>

#include "tree.h"
#include "data.h"


decision_tree * get_symmetric_dense(std::size_t height, val_type value, int (*get_idx)(std::size_t)) {
    auto r = 33;
    if (height == 0) {
        auto leaf = new decision_tree(value, r);
        return leaf;
    }

    auto left_son = get_symmetric_dense(height -1, value, get_idx);
    auto right_son = get_symmetric_dense(height -1, value, get_idx);
    return new decision_tree(0.0, left_son->prob +right_son->prob, true, get_idx(height), left_son, right_son);
}



decision_tree * get_symmetric_sparse(std::size_t height, val_type value, int (*get_idx)(std::size_t)) {
    auto r = 33;
    auto leaf = new decision_tree(value, r);
    if (height == 0) {
        return leaf;
    }

    auto left_son = leaf;
    auto right_son = get_symmetric_sparse(height -1, value, get_idx);
    return new decision_tree(0.0, left_son->prob +right_son->prob, true, get_idx(height), left_son, right_son);
}


int get_height_idx(std::size_t height) {
    return height - 1;
}

int get_next_idx(std::size_t height) {
    static int cnt = 0;
    return cnt++;
}

template <class GetTree>
decision_tree * get_symmetric_0_1(std::size_t height, GetTree get_tree, int (*get_idx)(std::size_t)) {
    auto left = get_tree(height -1, 0, get_idx);
    auto right = get_tree(height -1, 777, get_idx);
    auto tree = new decision_tree(0.0, left->prob + right->prob, true, get_idx(height), left, right);
    tree->compute_unique_height();
    return tree;
}

Data get_symmetric_data(std::size_t height) {
    return Data{Row(height, 1)};
}

Data get_symmetric_data_all(std::size_t height) {
    if (height == 1) {
        return Data{{1}, {-1}};
    }

    auto data_prev = get_symmetric_data_all(height-1);
    Data ret;
    ret.reserve(data_prev.size()*2); // preallocate memory
    ret.insert(ret.end(), data_prev.begin(), data_prev.end());
    ret.insert(ret.end(), data_prev.begin(), data_prev.end());
    for (std::size_t i = 0; i < ret.size(); ++i) {
        ret[i].push_back(1?-1:i<data_prev.size());
    }
    return ret;
}
