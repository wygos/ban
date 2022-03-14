#ifndef BAN_SHAP_COEFF_H
#define BAN_SHAP_COEFF_H

#include <numeric>

#include "common.h"

// the two following structs are used only for the brute force computation

// for shap the weight is 1 / (n * binom(n - 1, k))
struct shap_subset_coeff {
  std::size_t n;
  res_row binom;
  
  shap_subset_coeff(std::size_t _n) : n(_n), binom(n) {
    binom[0] = 1;
    for (std::size_t i = 1; i < n; ++i) {
      for (std::size_t j = i; j >= 1; --j) {
        binom[j] += binom[std::size_t(j - 1)];
      }
    }
  }

  res_type operator()(std::size_t k) const {
    return n * binom[k];
  }
};

// for banzhaf we ignore the size of the subset and return 1 / (2 ^ (n-1))
struct banzhaf_subset_coeff {
  int n;

  banzhaf_subset_coeff(int _n) : n(_n) {}

  res_type operator()(int k) const {
    return 1 << (n - 1);
  }
};

// The two following structs represent the unified DP state for simple and fast
// implementation. The state's parameters are n ``feature coefficients'' for all
// the unique features on the root -- v path for some node v.
// The state can be extended by adding a new feature (when moving to a child node),
// or by removing a feature (which basicaly cancels the insertion).

// shap DP state has n values, where n is the number of features. additions and 
// deletions of features take O(n) time.
struct shap_state {
  std::size_t n;
  res_row dp;

  shap_state(std::size_t max_features) : n(0), dp(max_features + 1, 0) {
    dp[0] = 1;
  }

  shap_state& operator=(const shap_state &s) {
    n = s.n;
    std::copy(s.dp.begin(), s.dp.begin() + n + 1, dp.begin());
    return *this;
  }

  void add_feature(res_type c) {
    ++n;
    //assert(n < int(dp.size()));
    for (std::size_t i = n; i >= 1; --i) {
      dp[i] = (dp[i] * std::size_t(n - i) + dp[std::size_t(i - 1)] * c * i) / (n + 1);
    }
    dp[0] *= res_type(n) / (n + 1);
  }

  void del_feature(res_type c) {
    dp[0] *= res_type(n + 1) / n;
    for (std::size_t i = 1; i < n; ++i) {
      dp[i] -= dp[i - 1] * (res_type(i) / (n + 1)) * c;
      dp[i] *= res_type(n + 1) / (n - i);
    }
    /*
    if (is_zero(c)) {
      for (std::size_t i = 1; i < n; ++i) {
        dp[i] *= res_type(n + 1) / std::size_t(n - i);
      }
    } else {
      res_type carry = dp[n];
      for (std::size_t i = n - 1; i > 0; --i) {
        res_type tmp = dp[i];
        dp[i] = (carry * (n + 1) - dp[i + 1] * (n - i - 1)) / (c * (i + 1));
        carry = tmp;
      }
    }*/
    --n;
  }

  res_type del_aggr(res_type c) {
    res_type res = 0;
    if (is_zero(c)) {
      for (std::size_t i = 0; i < n; ++i) {
        res += dp[i] * (n + 1) / (n - i);
      }
    } else {
      res_type cur = dp[n];
      for (int i = n - 1; i >= 0; --i) {
        cur = (dp[i + 1] * (n + 1) - cur * (n - i - 1)) / (c * (i + 1));
        res += cur;
      }
    }
    return res;
  }

  void scale(res_type c) {
    for (auto & i:dp) {
      i *= c;
    }
  }

  // sum through all subset sizes
  res_type aggr() const {
    return std::accumulate(dp.begin(), dp.begin() + int(n + 1), res_type(0));
  }

  // arithmetic operations on the dp
  shap_state& operator+=(const shap_state &d) {
    //assert(n == d.n);
    for (std::size_t i =0; i <= n; ++i) {
      dp[i] += d.dp[i];
    }
    return *this;
  }

  shap_state& operator-=(const shap_state &d) {
    //assert(n == d.n);
    for (std::size_t i =0; i <= n; ++i) {
      dp[i] -= d.dp[i];
    }
    return *this;
  }
};

// Banzhaf state consists of a single value. The additions and deletions take O(1) time.
struct banzhaf_state {
  res_type dp;

  banzhaf_state(std::size_t  max_features) : dp(1) {}

  void add_feature(res_type c) {
    dp = 0.5 * (1 + c) * dp;
  }

  void del_feature(res_type c) {
    dp = 2 * dp / (1 + c);
  }

  void scale(res_type c) {
    dp *= c;
  }

  res_type aggr() const {
    return dp;
  }

  res_type del_aggr(res_type c) {
    return 2 * dp / (1 + c);
  }

  banzhaf_state& operator+=(const banzhaf_state &d) {
    dp += d.dp;
    return *this;
  }

  banzhaf_state& operator-=(const banzhaf_state &d) {
    dp -= d.dp;
    return *this;
  }
};

// a helper function computing a coefficient of a feature based on the tree path
res_type coef(const Row &x,std::vector<bool> const & missing_reach, std::vector<interval> const & iv, res_row const & prob, std::size_t f) {
  if (std::isnan(x[f])) {
    return ((bool)missing_reach[f]) / prob[f];
  } else {
    return iv[f].contains(x[f]) / prob[f];
  }
}

#endif

