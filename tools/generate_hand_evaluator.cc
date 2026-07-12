#include "tools/hand_evaluator_table_builder.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <utility>
#include <vector>

namespace poker {
namespace {

using hand_evaluator_generation::TableData;

void PrintUint16Array(const char* name,
                      const std::array<uint16_t, 8192>& values) {
  std::cout << "inline constexpr std::array<uint16_t, 8192> " << name
            << " = {{\n";
  for (size_t i = 0; i < values.size(); ++i) {
    const bool line_end = i % 12 == 11 || i + 1 == values.size();
    if (i % 12 == 0) {
      std::cout << "    ";
    }
    std::cout << values[i];
    if (i + 1 != values.size()) {
      std::cout << ",";
    }
    if (line_end) {
      std::cout << "\n";
    } else {
      std::cout << " ";
    }
  }
  std::cout << "}};\n\n";
}

void PrintProducts(const std::vector<std::pair<int, uint16_t>>& products) {
  std::cout << "inline constexpr std::array<std::pair<int, uint16_t>, "
            << products.size() << "> kCactusProducts = {{\n";
  for (const auto& [product, value] : products) {
    std::cout << "    {" << product << ", " << value << "},\n";
  }
  std::cout << "}};\n\n";
}

void PrintHeader(const TableData& tables) {
  std::cout << "#pragma once\n\n"
            << "#include <array>\n"
            << "#include <cstdint>\n"
            << "#include <utility>\n\n"
            << "namespace poker::hand_evaluator_tables {\n\n";
  PrintUint16Array("kCactusFlushes", tables.flushes);
  PrintUint16Array("kCactusUnique5", tables.unique5);
  PrintProducts(tables.products);
  std::cout << "}  // namespace poker::hand_evaluator_tables\n";
}

}  // namespace
}  // namespace poker

int main() {
  poker::PrintHeader(poker::hand_evaluator_generation::BuildCactusTables());
  return 0;
}
