#include "tree.h"
#include "data.h"


template<typename dp_state>
std::vector<res_row> compute_brute(Trees const & Ts,
    const Data &data) {
  std::vector<res_row> res;
  for(auto const & x: data) {
    res.push_back(compute_brute<dp_state>(Ts, x));
  }
  return res;
}



res_type  eval(decision_tree *v, int msk, Row const & x){
  if (v->is_leaf()) {
    return v->t;
  }
  if (msk & (1 << v->feature)) {
    if (negative(x[v->feature] - v->t)) {
      return eval(v->left, msk, x);
    } else {
      return eval(v->right, msk, x);
    }
  }
  assert(!is_zero(v->prob));
  return (v->left->prob * eval(v->left, msk, x) + v->right->prob * eval(v->right, msk, x))/v->prob;
};

res_type  eval(Trees const & Ts, int msk, Row const & x){
  res_type res = 0.;
  for (decision_tree *T : Ts) {
    res += eval(T, msk, x);
  }
  return res;
}

// A brute force computation
// subset_coeff(N) is supposed to return, given k, a weight of a k-element subset of [N]
template<typename subset_coeff>
res_row compute_brute(Trees const & Ts, const Row &x) {
  auto N = int(x.size());
  assert(N <= 30);
  res_row res((std::size_t)N);
  subset_coeff coeff(N);
  for (int g = 0; g < N; ++g) {
      for (int mask = 0; mask < (1 << N); ++mask) {
        if (mask & (1 << g)) {
          continue;
        }
        int subset_size = __builtin_popcount((unsigned int)(mask));
        res[g] += (eval(Ts, mask | (1 << g), x) - eval(Ts, mask, x)) / coeff(subset_size);
      }
    }
  return res;
}


template<typename subset_coeff>
std::vector<res_row> compute_brute(int N, Trees const & Ts, Data const &data) {
  std::vector<res_row> res;
  for(auto const & x: data) {
    res.push_back(compute_brute<subset_coeff>(N, Ts, x));
  }
  return res;
}
