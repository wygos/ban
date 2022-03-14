#include <cmath>
#include "coeff.h"

// computation of shapley/banzhaf values
template<typename dp_state>
struct stable {
  // the input data
  const Trees &Ts;
  const Data &data;
  // the number of features
  std::size_t N;
  // auxiliary data structures
  // number of times a given feature appears on the current path
  std::vector<int> feature_cnt;
  // the interval that a given feature's value has to be in for the current path to arise
  std::vector<interval> iv;
  // if the feature is missing, could we end up in the current path?
  std::vector<bool> missing_reach;
  // the probability of the current path distributed through the current path's features
  res_row prob;
  
  // after the computation is over, these are the shap/banzhaf values
  res_row res;
  // the list of current path's features
  std::vector<std::size_t> current_features;

  const Row *x;
  int __ch;

  stable(const Trees &_Ts, const Data &_data) : Ts(_Ts), data(_data), N(data.back().size()),
    feature_cnt(N, 0), iv(N), missing_reach(N, true), prob(N, 1) {
    int mx_h = 0;
    for (decision_tree *t : Ts) {
      mx_h = std::max(mx_h, (int)t->height());
    }
  }

  std::vector<res_row> compute() {
    std::vector<res_row> ret;
    for(auto const &_x: data) {
      // for each data point
      x = &_x;
      // reset the shap/banzhaf values vector
      res.assign(N, 0);
      // process each tree
      for (decision_tree *t : Ts) {
        if (t->is_leaf()) {
          continue;
        }
        // reset the dp state
        __ch = t->unique_height;
        traverse(t->left, t);
        traverse(t->right, t);
      }
      // add to the results
      ret.push_back(res);
    }
    return ret;
  }

  void traverse(decision_tree *v, decision_tree *par) {
    auto f = par->feature;
    // update the current features
    // if the feature was not on the path before, add it
    if (feature_cnt[f] == 0) {
      current_features.push_back(f);
    }
    // update the auxiliary data
    ++feature_cnt[f];
    prob[f] *= v->prob;
    interval old_iv = iv[f];
    bool old_missing_reach = missing_reach[f];
    if (par->left == v) {
      iv[f] &= interval(interval::minus_inf(), par->t);
      missing_reach[f] = missing_reach[f] && par->missing_go_left;
    } else {
      iv[f] &= interval(par->t);
      missing_reach[f] = missing_reach[f] && !par->missing_go_left;
    }
    if (v->is_leaf()) {
      // for each feature on the path, produce a state without that feature
      // if there are F features, this works in O(F log F * dp_state op time)
      std::function<void(int, int, dp_state&)> rec = [&](int xx, int yy, dp_state &d) -> void {
        if (xx == yy) {
          auto b = current_features[xx];
          d.scale(prob[b]);
          res_type c = coef(*x, missing_reach, iv, prob, b);
          res[b] += d.aggr() * (c - 1) * v->t;
          return;
        }
        int mid = (xx + yy) / 2;
        dp_state tmp = d;
        for (int i = xx; i <= mid; ++i) {
          auto f = current_features[i];
          tmp.scale(prob[f]);
          tmp.add_feature(coef(*x, missing_reach, iv, prob, f));
        }
        rec(mid + 1, yy, tmp);
        tmp = d;
        for (int i = mid + 1; i <= yy; ++i) {
          auto f = current_features[i];
          tmp.scale(prob[f]);
          tmp.add_feature(coef(*x, missing_reach, iv, prob, f));
        }
        rec(xx, mid, tmp);
      };
      dp_state tmp(__ch);
      rec(0, int(current_features.size()) - 1, tmp);
    } else {
      // otherwise process the two branches recursively
      traverse(v->left, v);
      traverse(v->right, v);
    }
    // revert all the updates, restore the state before this call
    prob[f] /= v->prob;
    iv[f] = old_iv;
    missing_reach[f] = old_missing_reach;
    --feature_cnt[f];
    if (feature_cnt[f] == 0) {
      current_features.pop_back();
    }
  };

};

template<typename dp_state>
std::vector<res_row> compute_stable(Trees const & Ts, const Data &data) {
  stable<dp_state> f(Ts, data);
  return f.compute();
}
