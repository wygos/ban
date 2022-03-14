#include "coeff.h"


template<typename dp_state>
struct fast {

  const Trees &Ts;
  const Data &data;
  std::size_t N;
  std::vector<int> feature_cnt;
  std::vector<interval> iv;
  std::vector<bool> missing_reach;
  res_row prob;
  dp_state dp;
  // after the children calls, Gs[f] is supposed to contain, for a feature f,
  // all nodes u such that 1) u has been processed,
  // 2) no processed ancestor of u has the same feature f
  std::vector<std::vector<int>> Gs;
  res_row res;
  const Row *x;
  int num, H, cur_h;
  std::vector<dp_state> dph;
  // the sums of dp states in the given subtree's leaves
  std::vector<dp_state> G;
  // the sums of dp state in the given subtree v's leaves with no intermediate ancestor
  // with the same feature
  std::vector<dp_state> S;

  fast(const Trees &_Ts, const Data &_data) : Ts(_Ts), data(_data), N(data.back().size()),
    feature_cnt(N, 0), iv(N), missing_reach(N, true), prob(N, 1), dp(N), Gs(N), cur_h(0) {
    int mx_size = 0, mx_h = 0, mx_H = 0;
    for (decision_tree *t : Ts) {
      mx_size = std::max(mx_size, t->size());
      mx_h = std::max(mx_h, (int)t->height());
      mx_H = std::max(mx_H, (int)t->unique_height);
    }
    dph = std::vector<dp_state>(mx_h, dp_state(N));
    G = std::vector<dp_state>(mx_size, dp_state(mx_H));
    S = std::vector<dp_state>(mx_size, dp_state(mx_H));
  }

  std::vector<res_row> compute() {
    std::vector<res_row> ret;
    for(auto const &_x: data) {
      x = &_x;
      res.assign(N, 0);
      for (decision_tree *t : Ts) {
        if (t->is_leaf()) {
            continue;
        }
        num = 0;
        H = t->unique_height;
        // initialize the dp state to H dummy features
        dp = dp_state(H);
        for (int i = 0; i < H; ++i) {
          dp.add_feature(1);
        }
        traverse(t->left, t);
        traverse(t->right, t);
      }
      ret.push_back(res);
    }
    return ret;
  }

  void traverse(decision_tree *v, decision_tree *par) {
    auto f = par->feature;
    dph[cur_h++] = dp;
    // the preorder number of v
    int pre = num++;
    if (feature_cnt[f] == 0) {
      // we assume the dp state to always contain H (possibly dummy) features
      dp.del_feature(1);
    } else {
      dp.del_feature(coef(*x, missing_reach, iv, prob, f));
    }
    ++feature_cnt[f];
    prob[f] *= v->prob;
    dp.scale(v->prob);
    interval old_iv = iv[f];
    bool old_missing_reach = missing_reach[f];
    if (par->left == v) {
      iv[f] &= interval(interval::minus_inf(), par->t);
      missing_reach[f] = missing_reach[f] && par->missing_go_left;
    } else {
      iv[f] &= interval(par->t);
      missing_reach[f] = missing_reach[f] && !par->missing_go_left;
    }
    res_type c = coef(*x, missing_reach, iv, prob, f);
    dp.add_feature(c);
    if (v->is_leaf()) {
      // the sum of dp's in the leaf is just the dp scaled by the leaf's weight
      G[pre] = dp;
      G[pre].scale(v->t);
    } else {
      // the sum of dp's in an internal node is the sum of these values for the children
      traverse(v->left, v);
      G[pre] = G[pre + 1];
      int pt = num;
      traverse(v->right, v);
      G[pre] += G[pt];
    }
    // first compute S_u' from G, clean up Gs
    S[pre] = G[pre];
    while (!Gs[f].empty() && Gs[f].back() >= pre) {
      S[pre] -= G[Gs[f].back()];
      Gs[f].pop_back();
    }
    res[f] += S[pre].del_aggr(c) * (c - 1);
    // only update Gs if some ancestor has the same feature
    if (feature_cnt[f] > 1) {
      Gs[f].push_back(pre);
    }
    // revert changes
    prob[f] /= v->prob;
    iv[f] = old_iv;
    missing_reach[f] = old_missing_reach;
    --feature_cnt[f];
    dp = dph[--cur_h];
  }

};

template<typename dp_state>
std::vector<res_row> compute_fast(Trees const & Ts, const Data &data) {
  fast<dp_state> f(Ts, data);
  return f.compute();
}
