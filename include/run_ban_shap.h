#include <iostream>
#include <fstream>
#include <cassert>
#include <set>
#include <vector>


#include "compute_brute.h"
#include "compute_simple.h"
#include "compute_fast.h"
#include "compute_stable.h"
#include "original.h"
#include "coeff.h"


void norm_probs_in_tree(decision_tree & t) {
  if(!t.is_leaf()) {
    auto s = t.left->prob + t.right->prob;
    assert (s > EPSILON);
    t.left->prob /= s;
    t.right->prob /= s;
    norm_probs_in_tree(*t.left);
    norm_probs_in_tree(*t.right);
  }
}

void repair_probs_in_trees(Trees & T, Data const & data) {
  for (auto const & tree: T) {
    tree->prob = 1;
    norm_probs_in_tree(*tree);
  }
}



void print_values(std::string const & fname, std::string const & header, const std::vector<res_row> &res, std::string label)  {
  assert (!label.empty());
  std::ofstream ofs (fname + "." + label, std::ofstream::out);
  std::cout << label << ": " << std::endl;
  ofs << header << std::endl;
  for (auto const & a: res) {
   ofs << a[0];
   for (std::size_t i = 1; i < a.size(); ++i) {
    ofs << "," << a[i];
   }
   ofs << std::endl;
  }
}



void run_ban_shap(std::string const & cmd, std::string const &  fname, std::string const &  header, Trees & T, Data const& X) {
  auto c = tolower(cmd[0]);

  //no normalization in the original algorithm
  if (c != 'o') {
    repair_probs_in_trees(T, X);
  }

  if (c == 's') {
//    print_values(fname, header, compute_brute<shap_subset_coeff>(T, X), "shap_brute");
    print_values(fname, header, compute_simple<shap_state>(T, X), "shap_simple");
    print_values(fname, header, compute_fast<shap_state>(T, X), "shap_fast");
  } else if (c == 'b') {
  //  print_values(fname, header, compute_brute<banzhaf_subset_coeff>(T, X), "banzhaf_brute");
    print_values(fname, header, compute_simple<banzhaf_state>(T, X), "banzhaf_simple");
    print_values(fname, header, compute_fast<banzhaf_state>(T, X), "banzhaf_fast");
  } else if (c == 'f') {
    print_values(fname, header, compute_fast<banzhaf_state>(T, X), "banzhaf_fast");
  } else if (c == 'i') {
    print_values(fname, header, compute_simple<banzhaf_state>(T, X), "banzhaf_simple");
  } else if (c == 'g') {
    print_values(fname, header, compute_simple<shap_state>(T, X), "shap_simple");
  } else if (c == 'h') {
    print_values(fname, header, compute_fast<shap_state>(T, X), "shap_fast");
  } else if (c == 'o') {
    print_values(fname, header, compute_original(T, X), "shap_orig_c");
  } else if (c == 'z') {
    print_values(fname, header, compute_stable<shap_state>(T, X), "shap_stable");
  } else {
    std::cerr << "Unknown command." << std::endl;
    exit(1);
  }
 }
