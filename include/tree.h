#ifndef BAN_SHAP_TREE_H
#define BAN_SHAP_TREE_H


#include "common.h"

#include <fstream>
#include <map>
#include <cassert>
#include <functional>
#include <iostream>


// representation of a decision tree
struct decision_tree {
  // threshold for internal nodes, value for leaves
  val_type t;
  bool missing_go_left;
  // only makes sense for an internal node
  std::size_t feature;
  decision_tree *left, *right;
  // probability of entering the node if parent was entered
  res_type prob;
  std::size_t unique_height;

  decision_tree(val_type _t, res_type _p = 0.0, bool _m = true, std::size_t f = std::size_t(-1), decision_tree *l = nullptr,
    decision_tree *r = nullptr)
    : t(_t), missing_go_left(_m), feature(f), left(l), right(r), prob(_p) {}

  bool is_leaf() const {
    return left == nullptr && right == nullptr;
  }


  //TODO compute in contructor?, remove static variable
  void compute_unique_height() {
    static std::set<std::size_t> current;
    if (is_leaf()) {
      unique_height = current.size();
    } else {
      bool was_there = current.contains(feature);
      if (!was_there) {
        current.insert(feature);
      }
      left->compute_unique_height();
      right->compute_unique_height();
      unique_height = std::max(left->unique_height, right->unique_height);
      if (!was_there) {
        current.erase(feature);
      }
    }
  }
      
  int height() const {
      if  (is_leaf()) {
          return 0;
      } else {
          return std::max(left->height(), right->height()) + 1;
      }
  }
  
  int size() const {
      if  (is_leaf()) {
          return 1;
      } else {
          return left->size() + right->size() + 1;
      }
  }

};

using Trees = std::vector<decision_tree*>;


// this is supposed to parse the xgboost dump
//   that use f to branch
void read_trees(Trees &res, std::map<std::string, std::size_t> const & id, std::ifstream& if_trees) {
  std::string tmp;
  std::function<decision_tree*(void)>  read_rec = [&](void) -> decision_tree* {
    getline(if_trees, tmp, ':');
    char type;
    res_type p;
    if_trees >> type;
    if (type == 'l') {
      getline(if_trees, tmp, '=');
      val_type leaf_val;
      if_trees >> leaf_val;
      getline(if_trees, tmp, '=');
      if_trees >> p;
      getline(if_trees, tmp);
      return new decision_tree(leaf_val, p);
    }
    assert(type == '[');
    std::string f;
    getline(if_trees, f, '<');
    val_type threshold;
    if_trees >> threshold;
    int yes, no, missing;
    getline(if_trees, tmp, '=');
    if_trees >> yes;
    getline(if_trees, tmp, '=');
    if_trees >> no;
    assert(no == yes + 1);
    getline(if_trees, tmp, '=');
    if_trees >> missing;
    assert(missing == yes || missing == no);
    getline(if_trees, tmp, '=');
    getline(if_trees, tmp, '=');

    if_trees >> p;
    getline(if_trees, tmp);
    decision_tree *left = read_rec();
    decision_tree *right = read_rec();
    return new decision_tree(threshold, p, missing == yes, id.at(f), left, right);
  };
  res.clear();
  while (getline(if_trees, tmp)) {
    res.push_back(read_rec());
    res.back()->compute_unique_height();
  }
}




// A.K. my internal testing format
std::size_t read_trees_test(Trees &res, std::vector<val_type> &X) {
  std::size_t N; std::cin >> N;
  std::string cmd;
  std::function<decision_tree*(void)> read_rec = [&](void) -> decision_tree* {
    std::cin >> cmd;
    if (cmd == "L") {
      val_type t;
      std::cin >> t;
      return new decision_tree(t);
    } else if (cmd == "V") {
      std::size_t f; std::cin >> f;
      val_type t, p1, p2; std::cin >> t >> p1 >> p2;
      assert(is_zero(p1 + p2 - 1));
      decision_tree *left = read_rec();
      decision_tree *right = read_rec();
      left->prob = p1;
      right->prob = p2;
      return new decision_tree(t, p1, true, f, left, right);
    } else {
      assert(false);
    }
  };
  res = {read_rec()};
  X = std::vector<val_type>(N);
  for (auto & x: X) {
    std::cin >> x;
  }
  return N;
}
#endif