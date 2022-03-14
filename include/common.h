#ifndef BAN_SHAP_COMMON_H
#define BAN_SHAP_COMMON_H

#include <vector>
#include <limits>
#include <algorithm>
//P.W. had problem with compilation
//#include <boost/multiprecision/cpp_bin_float.hpp>
//#include <boost/multiprecision/float128.hpp>

using val_type = double;
using res_type = double;
//using res_type = boost::multiprecision::float128;
using res_row = std::vector<res_type>;

const res_type EPSILON = 1e-9;

bool is_zero(res_type x) {
  return x >= -EPSILON && x <= EPSILON;
}

bool negative(res_type x) {
  return x < -EPSILON;
}

// operations on intervals
struct interval {
  val_type a, b;

  static constexpr val_type minus_inf() {
    return std::numeric_limits<val_type>::lowest() / 2;
  }

  static constexpr val_type plus_inf() {
    return std::numeric_limits<val_type>::max() / 2;
  }

  interval(val_type _a = minus_inf(), val_type _b = plus_inf()) : a(_a), b(_b) {}

  bool contains(val_type x) const {
    return x - a >= -EPSILON && b - x > EPSILON;
  }

  interval& operator&=(const interval &x) {
    a = std::max(a, x.a);
    b = std::min(b, x.b);
    return *this;
  }
};


#endif
