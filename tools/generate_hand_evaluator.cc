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

const char* HandRankName(HandRank rank) {
  switch (rank) {
    case HandRank::HighCard:
      return "HandRank::HighCard";
    case HandRank::Pair:
      return "HandRank::Pair";
    case HandRank::TwoPair:
      return "HandRank::TwoPair";
    case HandRank::ThreeOfAKind:
      return "HandRank::ThreeOfAKind";
    case HandRank::Straight:
      return "HandRank::Straight";
    case HandRank::Flush:
      return "HandRank::Flush";
    case HandRank::FullHouse:
      return "HandRank::FullHouse";
    case HandRank::FourOfAKind:
      return "HandRank::FourOfAKind";
    case HandRank::StraightFlush:
      return "HandRank::StraightFlush";
    case HandRank::RoyalFlush:
      return "HandRank::RoyalFlush";
  }
}

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

void PrintScores(
    const std::array<hand_evaluator_generation::EvaluationScore,
                     hand_evaluator_generation::kCactusScoreCount>& scores) {
  std::cout << "inline constexpr std::array<ScoreRecord, "
            << scores.size() << "> kCactusScores = {{\n";
  for (const auto& score : scores) {
    std::cout << "    {" << HandRankName(score.rank) << ", {"
              << score.kickers[0] << ", " << score.kickers[1] << ", "
              << score.kickers[2] << ", " << score.kickers[3] << ", "
              << score.kickers[4] << "}, " << score.kicker_count << "},\n";
  }
  std::cout << "}};\n\n";
}

void PrintHeader(const TableData& tables) {
  std::cout << "#pragma once\n\n"
            << "#include <array>\n"
            << "#include <cstddef>\n"
            << "#include <cstdint>\n"
            << "#include <utility>\n\n"
            << "#include \"src/hand_evaluator.h\"\n\n"
            << "namespace poker::hand_evaluator_tables {\n\n"
            << "struct ScoreRecord {\n"
            << "  HandRank rank = HandRank::HighCard;\n"
            << "  std::array<int, 5> kickers = {};\n"
            << "  size_t kicker_count = 0;\n"
            << "};\n\n"
            << "inline constexpr size_t kCactusScoreCount = "
            << hand_evaluator_generation::kCactusScoreCount << ";\n\n";
  PrintUint16Array("kCactusFlushes", tables.flushes);
  PrintUint16Array("kCactusUnique5", tables.unique5);
  PrintProducts(tables.products);
  PrintScores(tables.scores);
  std::cout << "}  // namespace poker::hand_evaluator_tables\n";
}

}  // namespace
}  // namespace poker

int main() {
  poker::PrintHeader(poker::hand_evaluator_generation::BuildCactusTables());
  return 0;
}
