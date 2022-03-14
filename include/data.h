#ifndef BAN_SHAP_DATA_H
#define BAN_SHAP_DATA_H
#include <fstream>
#include <cmath>

#include "common.h"

using Row = std::vector<val_type>;
using Data = std::vector<Row>;

std::vector<std::string> split(std::string line) {
    std::string const delimiter = ",";
    std::vector<std::string> row;
    size_t pos = 0;
    std::string token;
    while ((pos = line.find(delimiter)) != std::string::npos)
    {
        token = line.substr(0, pos);
        row.push_back(token);
        line.erase(0, pos + delimiter.length());
    }
    row.push_back(line);

    return row;
}

std::string read_data(Data &X, std::map<std::string, std::size_t> & f_ids, std::ifstream& if_data) {
  bool firstLine = true;
  bool error = false;
  std::string header = "";
  if (if_data.good()) {
        std::string line;

        while (!error && std::getline(if_data, line)) {
            if (line.empty()) {
                error = true;
            } else {
               if(firstLine) {
                 firstLine = false;
                 header = line;
                 auto row_string = split(line);
                 for(std::size_t i =0; i < row_string.size(); ++i) {
                     f_ids[row_string[i]] = i;
                 }
               } else {
                 Row row;
                 auto row_string = split(line);
                 for(auto const & s: row_string){
                     if (s.length() == 0) {
                         row.push_back(std::nan(""));
                     } else {
                         row.push_back(std::stod(s));
                     }
                 }
                 X.push_back(row);
               }
            }
        }
  }
  assert(X.size() > 0);

  return header;
}
#endif