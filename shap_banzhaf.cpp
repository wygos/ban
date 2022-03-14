#include <iomanip>

#include "include/run_ban_shap.h"


// shap and banzhaf computed using three different algorithms
// change the "s" to "b" to get banzhaf values instead
// test = true is something I used for testing
void compute(std::string cmd, std::ifstream & if_trees, std::ifstream & if_data, std::string fname, bool test = false) {
  Trees T;
  Data X;
  std::map<std::string, std::size_t> f_ids;
  auto header = read_data(X, f_ids, if_data);
  read_trees(T, f_ids, if_trees);

  std::cout << "data read" << std::endl;
  run_ban_shap(cmd, fname, header, T, X);
}


int main(int argc, char** argv) {
  if (argc < 4) {
    std::cout << "usage:" << std::endl;
    std::cout << "all shap ours   :" << "\t" <<  argv[0] << " s bst_boston.file boston.csv" << std::endl;
    std::cout << "all banzhaf:" << "\t" <<  argv[0] << " b bst_boston.file boston.csv" << std::endl;
    std::cout << "banzhaf fast:" << "\t" <<  argv[0] << " f bst_boston.file boston.csv" << std::endl;
    std::cout << "banzhaf simple:" << "\t" <<  argv[0] << " i bst_boston.file boston.csv" << std::endl;
    std::cout << "shap ours simple:" << "\t" <<  argv[0] << " g bst_boston.file boston.csv" << std::endl;
    std::cout << "shap ours fast:" << "\t" <<  argv[0] << " h bst_boston.file boston.csv" << std::endl;
    std::cout << "shap form original library:" << "\t./" <<  argv[0] << " o" << std::endl;
    std::cout << "shap stable:" << "\t./" <<  argv[0] << " z" << std::endl;
    return -1;
  }

  std::ifstream if_trees (argv[2], std::ifstream::in);
  std::ifstream if_data (argv[3], std::ifstream::in);

  compute(argv[1], if_trees, if_data, argv[2], argc > 2);
  return 0;
}

