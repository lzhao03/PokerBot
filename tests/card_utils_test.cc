#include "src/card_abstraction.h"
#include "src/hand_evaluator.h"
#include "src/poker.h"
#include "src/solver.h"
#include "tools/hand_evaluator_table_builder.h"

#include "doctest/doctest.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <initializer_list>
#include <random>
#include <set>
#include <stdexcept>

namespace poker {
namespace {

using S = Suit;

Card C(int rank, S suit) { return Card(static_cast<Rank>(rank - 2), suit); }
ComboId H(Card a, Card b) { return CardsToComboId(a, b); }

Board Runout(absl::Span<const Card> cards) {
  if (cards.empty()) {
    return PreflopBoard{};
  }
  const std::array<Card, 3> flop_cards = {cards[0], cards[1], cards[2]};
  const FlopBoard flop = DealFlop(PreflopBoard{}, flop_cards);
  if (cards.size() == 3) {
    return flop;
  }
  const TurnBoard turn = DealTurn(flop, cards[3]);
  if (cards.size() == 5) {
    return DealRiver(turn, cards[4]);
  }
  return turn;
}

Board B(std::initializer_list<Card> cards) {
  return Runout(absl::Span<const Card>(cards.begin(), cards.size()));
}

EquityBucketModel TestEquityModel() {
  EquityBucketModel model;
  model.rollout_seed = 7;
  model.fit_seed = 11;
  model.training_samples = 100;
  model.opponent_samples = 16;
  model.runout_samples = 8;
  for (StreetKind street : {StreetKind::kFlop, StreetKind::kTurn}) {
    auto& cutoffs = model.ehs2_cutoffs[static_cast<size_t>(street)];
    auto& medians = model.ehs_medians[static_cast<size_t>(street)];
    for (int index = 1; index < 16; ++index) {
      cutoffs.push_back(static_cast<float>(index) / 16.0f);
    }
    medians.assign(16, 0.5f);
  }
  for (int index = 1; index < 64; ++index) {
    model.river_equity_cutoffs.push_back(
        static_cast<float>(index) / 64.0f);
  }
  const auto finalized = FinalizeEquityBucketModel(std::move(model));
  if (!finalized.ok()) {
    throw std::invalid_argument(std::string(finalized.status().message()));
  }
  return *finalized;
}

HandEvaluation Score5(const std::array<Card, 5>& cards) {
  const auto score = hand_evaluator_generation::EvaluateFiveCardScore(cards);
  HandEvaluation out;
  out.rank = score.rank;
  out.kickers = score.kickers;
  out.kicker_count = score.kicker_count;
  return out;
}

HandEvaluation ReferenceBest(ComboId hand, const Board& board) {
  std::array<Card, 7> cards = {};
  const ComboInfo& combo = GetComboInfo(hand);
  cards[0] = combo.card0;
  cards[1] = combo.card1;
  size_t count = 2;
  for (Card card : BoardCards(board)) cards[count++] = card;

  HandEvaluation best;
  for (size_t a = 0; a + 4 < count; ++a)
    for (size_t b = a + 1; b + 3 < count; ++b)
      for (size_t c = b + 1; c + 2 < count; ++c)
        for (size_t d = c + 1; d + 1 < count; ++d)
          for (size_t e = d + 1; e < count; ++e)
            best = std::max(best, Score5({cards[a], cards[b], cards[c],
                                          cards[d], cards[e]}));
  return best;
}

int ReferenceCompare(ComboId a, ComboId b, const Board& board) {
  const HandEvaluation ea = ReferenceBest(a, board);
  const HandEvaluation eb = ReferenceBest(b, board);
  return ea > eb ? 1 : ea < eb ? -1 : 0;
}

Card Rename(Card card, const std::array<S, 4>& suits) {
  return C(PokerRank(card),
           suits[static_cast<size_t>(SuitIndex(CardSuit(card)))]);
}

Board Rename(const Board& board, const std::array<S, 4>& suits) {
  std::array<Card, kMaxBoardCards> cards = {};
  const auto board_cards = BoardCards(board);
  for (size_t i = 0; i < board_cards.size(); ++i) {
    cards[i] = Rename(board_cards[i], suits);
  }
  return Runout(absl::Span<const Card>(cards.data(), BoardCount(board)));
}

ComboId Rename(ComboId hand, const std::array<S, 4>& suits) {
  const ComboInfo& combo = GetComboInfo(hand);
  return H(Rename(combo.card0, suits), Rename(combo.card1, suits));
}

TEST_CASE("combo ids are an exhaustive canonical bijection") {
  size_t expected = 0;
  for (int a = 0; a < kDeckCardCount; ++a) {
    for (int b = a + 1; b < kDeckCardCount; ++b, ++expected) {
      const Card ca = kDeck[static_cast<size_t>(a)];
      const Card cb = kDeck[static_cast<size_t>(b)];
      CHECK(H(ca, cb).index() == expected);
      CHECK(H(cb, ca).index() == expected);
      const ComboInfo& info = GetComboInfo(H(ca, cb));
      CHECK(info.card0 == ca);
      CHECK(info.card1 == cb);
      CHECK(info.mask == (CardBit(ca) | CardBit(cb)));
    }
  }
  CHECK(expected == kComboCount);
  CHECK_FALSE(MakeHoleCards(kDeck[0], kDeck[0]).ok());
  const auto hole_cards = MakeHoleCards(kDeck[0], kDeck[1]);
  REQUIRE(hole_cards.ok());
  CHECK(hole_cards->combo() == H(kDeck[0], kDeck[1]));
}

TEST_CASE("hand evaluator matches an independent five-card oracle") {
  const auto wheel = EvaluateFiveCards(std::array<Card, 5>{
      C(14, S::kHearts), C(5, S::kDiamonds), C(4, S::kClubs),
      C(3, S::kSpades), C(2, S::kHearts)});
  CHECK(wheel.rank == HandRank::STRAIGHT);
  CHECK(wheel.kickers[0] == 5);

  std::array<Card, kDeckCardCount> deck = {};
  std::copy(kDeck.begin(), kDeck.end(), deck.begin());
  std::mt19937 rng(12345);
  for (int trial = 0; trial < 400; ++trial) {
    std::shuffle(deck.begin(), deck.end(), rng);
    const ComboId a = H(deck[0], deck[1]);
    const ComboId b = H(deck[2], deck[3]);
    const Board board = B({deck[4], deck[5], deck[6], deck[7], deck[8]});
    CAPTURE(trial);
    CHECK(CompareHands(a, b, std::get<RiverBoard>(board)) ==
          ReferenceCompare(a, b, board));
  }
}

TEST_CASE("sampling and card abstractions preserve identity") {
  std::array<Card, kDeckCardCount> deck = {};
  std::copy(kDeck.begin(), kDeck.end(), deck.begin());
  std::mt19937 rng(7);

  for (int trial = 0; trial < 150; ++trial) {
    std::shuffle(deck.begin(), deck.end(), rng);
    const StreetKind street = static_cast<StreetKind>(trial % 3);
    const int board_count = BoardCardsForStreet(street);
    const Board board = Runout(
        absl::Span<const Card>(deck.data(), board_count));
    const ComboId hand = H(deck[board_count], deck[board_count + 1]);

    std::array<Card, kMaxBoardCards> permuted = {};
    const auto cards = BoardCards(board);
    std::copy(cards.begin(), cards.end(), permuted.begin());
    if (BoardCount(board) >= 3) {
      std::swap(permuted[0], permuted[2]);
    }
    const Board reversed = Runout(
        absl::Span<const Card>(permuted.data(), BoardCount(board)));
    const BoardFeatures features = BoardFeaturesFor(board);
    CHECK(BoardFeaturesFor(reversed) == features);
    CHECK(BoardTextureBucket(street, features) ==
          BoardTextureBucket(street, BoardFeaturesFor(reversed)));
    CHECK(CoarsePrivateBucket(hand, street, features) <
          kCoarsePrivateStreetObservationCount);

    std::array<S, 4> suits = {
        S::kHearts, S::kDiamonds, S::kClubs, S::kSpades};
    std::shuffle(suits.begin(), suits.end(), rng);
    const Board renamed_board = Rename(board, suits);
    CHECK(BoardTextureBucket(street, features) ==
          BoardTextureBucket(street, BoardFeaturesFor(renamed_board)));

    const auto sampled_result =
        SampleStreetCards(street, board, ComboMask(hand), rng);
    REQUIRE(sampled_result.ok());
    const auto& sampled = *sampled_result;
    CHECK(sampled.size() == static_cast<size_t>(CardsForNextStreet(street)));
    CardMask blocked = BoardMask(board) | ComboMask(hand);
    for (Card card : sampled) {
      CHECK((blocked & CardBit(card)) == 0);
      blocked |= CardBit(card);
    }
  }

  CardMask blocked = 0;
  for (int i = 0; i < kDeckCardCount - 2; ++i)
    blocked |= CardBit(kDeck[static_cast<size_t>(i)]);
  CHECK_FALSE(SampleStreetCards(StreetKind::kPreflop,
                                Board{PreflopBoard{}}, blocked, rng).ok());
}

TEST_CASE("exact card observations are invariant under suit renaming") {
  const Board board = B({C(14, S::kHearts), C(13, S::kHearts),
                         C(7, S::kClubs), C(10, S::kDiamonds)});
  const ComboId hand = H(C(12, S::kHearts), C(11, S::kSpades));
  const CanonicalCardObservation expected =
      CanonicalizeObservation(hand, board);

  std::array<S, 4> suits = {
      S::kHearts, S::kDiamonds, S::kClubs, S::kSpades};
  do {
    const Board renamed_board = Rename(board, suits);
    CHECK(CanonicalPublicObservation(renamed_board) ==
          expected.public_observation);
    CHECK(CanonicalizeObservation(Rename(hand, suits), renamed_board) ==
          expected);
  } while (std::next_permutation(suits.begin(), suits.end()));

  const ComboId other_hand = H(C(9, S::kDiamonds), C(8, S::kSpades));
  CHECK(CanonicalizeObservation(other_hand, board).public_observation ==
        expected.public_observation);
}

TEST_CASE("handcrafted 36 mappings remain stable") {
  CHECK(CoarsePrivateBucket(
            H(C(14, S::kHearts), C(14, S::kSpades)),
            StreetKind::kPreflop, BoardFeatures{}) == 0);
  CHECK(CoarsePrivateBucket(
            H(C(14, S::kHearts), C(13, S::kHearts)),
            StreetKind::kPreflop, BoardFeatures{}) == 12);
  CHECK(CoarsePrivateBucket(
            H(C(7, S::kHearts), C(2, S::kSpades)),
            StreetKind::kPreflop, BoardFeatures{}) == 35);

  const Board flop = B({C(2, S::kHearts), C(7, S::kHearts),
                        C(12, S::kHearts)});
  const ComboId hand = H(C(14, S::kHearts), C(13, S::kSpades));
  CHECK(CoarsePrivateBucket(hand, StreetKind::kFlop,
                            BoardFeaturesFor(flop)) == 6);

  const CardAbstractionConfig current{
      PublicCardMode::kTexture,
      PrivateAbstractionKind::kHandcrafted36,
      RecallMode::kCurrentBucketOnly};
  const PublicPosition position =
      PublicPosition::Root(current, StreetKind::kFlop, flop);
  CHECK(ObservePrivate(current, hand, position) == PrivateObservationId(7));
}

TEST_CASE("equity models preserve cutoffs and bucket boundaries") {
  const EquityBucketModel model = TestEquityModel();
  CHECK(EquityBucket(StreetKind::kFlop, {0.49f, 0.0f}, model) == 0);
  CHECK(EquityBucket(StreetKind::kFlop, {0.50f, 0.0f}, model) == 1);
  CHECK(EquityBucket(StreetKind::kTurn, {0.49f, 1.0f}, model) == 30);
  CHECK(EquityBucket(StreetKind::kTurn, {0.50f, 1.0f}, model) == 31);
  CHECK(EquityBucket(StreetKind::kRiver, {0.0f, 0.0f}, model) == 0);
  CHECK(EquityBucket(StreetKind::kRiver, {1.0f, 1.0f}, model) == 63);

  const char* test_tmpdir = std::getenv("TEST_TMPDIR");
  REQUIRE(test_tmpdir != nullptr);
  const std::filesystem::path path =
      std::filesystem::path(test_tmpdir) / "equity.model";
  REQUIRE(SaveEquityBucketModel(model, path).ok());
  const auto loaded = LoadEquityBucketModel(path);
  REQUIRE(loaded.ok());
  CHECK(*loaded == model);

  EquityBucketModel changed = model;
  changed.river_equity_cutoffs[0] += 0.001f;
  CHECK_FALSE(ValidateEquityBucketModel(changed).ok());
}

TEST_CASE("canonical observations preserve card relationships and order") {
  const Board monotone = B({C(14, S::kHearts), C(13, S::kHearts),
                            C(12, S::kHearts)});
  const Board rainbow = B({C(14, S::kHearts), C(13, S::kDiamonds),
                           C(12, S::kClubs)});
  CHECK(CanonicalPublicObservation(monotone) !=
        CanonicalPublicObservation(rainbow));

  const Board draw_board = B({C(2, S::kHearts), C(7, S::kHearts),
                              C(12, S::kClubs)});
  const ComboId flush_draw = H(C(14, S::kHearts), C(13, S::kSpades));
  const ComboId no_flush_draw =
      H(C(14, S::kDiamonds), C(13, S::kSpades));
  CHECK(CanonicalizeObservation(flush_draw, draw_board) !=
        CanonicalizeObservation(no_flush_draw, draw_board));

  const Board jack_then_ten =
      B({C(14, S::kHearts), C(13, S::kDiamonds), C(12, S::kClubs),
         C(11, S::kSpades), C(10, S::kHearts)});
  const Board ten_then_jack =
      B({C(14, S::kHearts), C(13, S::kDiamonds), C(12, S::kClubs),
         C(10, S::kHearts), C(11, S::kSpades)});
  CHECK(BoardMask(jack_then_ten) == BoardMask(ten_then_jack));
  CHECK(CanonicalPublicObservation(jack_then_ten) !=
        CanonicalPublicObservation(ten_then_jack));
}

TEST_CASE("canonical observation counts match holdem suit isomorphisms") {
  std::set<uint64_t> private_observations;
  const Board preflop = PreflopBoard{};
  for (size_t first = 0; first < kDeck.size(); ++first) {
    for (size_t second = first + 1; second < kDeck.size(); ++second) {
      const ComboId hand = H(kDeck[first], kDeck[second]);
      private_observations.insert(
          CanonicalizeObservation(hand, preflop).private_observation.value());
    }
  }
  CHECK(private_observations.size() == 169);

  std::set<uint64_t> public_observations;
  for (size_t first = 0; first < kDeck.size(); ++first) {
    for (size_t second = first + 1; second < kDeck.size(); ++second) {
      for (size_t third = second + 1; third < kDeck.size(); ++third) {
        const FlopBoard flop = DealFlop(
            PreflopBoard{}, {kDeck[first], kDeck[second], kDeck[third]});
        public_observations.insert(
            CanonicalPublicObservation(Board{flop}).value());
      }
    }
  }
  CHECK(public_observations.size() == 1755);
}

TEST_CASE("all abstraction modes preserve observation history") {
  const std::array<CardAbstractionConfig, 4> configs = {{
      {PublicCardMode::kExactCanonical,
       PrivateAbstractionKind::kExactCanonical},
      {PublicCardMode::kExactCanonical,
       PrivateAbstractionKind::kHandcrafted36},
      {PublicCardMode::kTexture,
       PrivateAbstractionKind::kExactCanonical},
      {PublicCardMode::kTexture,
       PrivateAbstractionKind::kHandcrafted36},
  }};
  const ComboId hand = H(C(14, S::kHearts), C(13, S::kSpades));
  const FlopBoard flop = DealFlop(
      PreflopBoard{},
      {C(2, S::kHearts), C(7, S::kHearts), C(12, S::kClubs)});
  const TurnBoard turn = DealTurn(flop, C(9, S::kDiamonds));
  const RiverBoard river = DealRiver(turn, C(4, S::kSpades));

  for (const CardAbstractionConfig& config : configs) {
    CAPTURE(static_cast<int>(config.public_mode));
    CAPTURE(static_cast<int>(config.private_kind));
    PublicPosition position = PublicPosition::Root(
        config, StreetKind::kPreflop, Board{PreflopBoard{}});
    PrivateObservationId private_id =
        InitialPrivateObservation(config, hand);

    for (const auto& [street, board] : {
             std::pair{StreetKind::kFlop, Board{flop}},
             std::pair{StreetKind::kTurn, Board{turn}},
             std::pair{StreetKind::kRiver, Board{river}},
         }) {
      position = position.after_chance(config, street, board);
      private_id =
          AdvancePrivateObservation(config, private_id, hand, position);
      CHECK(position.observation() == ObservePublic(config, street, board));
      CHECK(private_id == ObservePrivate(config, hand, position));
    }

    const std::array<S, 4> renamed_suits = {
        S::kClubs, S::kSpades, S::kHearts, S::kDiamonds};
    const Board renamed_board = Rename(Board{river}, renamed_suits);
    const ComboId renamed_hand = Rename(hand, renamed_suits);
    const PublicPosition renamed = PublicPosition::Root(
        config, StreetKind::kRiver, renamed_board);
    CHECK(renamed.observation() == position.observation());
    CHECK(ObservePrivate(config, renamed_hand, renamed) == private_id);
  }
}

TEST_CASE("coarse public exact private keeps relative flush information") {
  const CardAbstractionConfig config{
      PublicCardMode::kTexture,
      PrivateAbstractionKind::kExactCanonical};
  const Board board = B({C(2, S::kHearts), C(7, S::kHearts),
                         C(12, S::kClubs)});
  const ComboId hand = H(C(14, S::kHearts), C(13, S::kSpades));
  const PublicPosition position =
      PublicPosition::Root(config, StreetKind::kFlop, board);

  const std::array<S, 4> suits = {
      S::kDiamonds, S::kClubs, S::kSpades, S::kHearts};
  const Board renamed_board = Rename(board, suits);
  const ComboId renamed_hand = Rename(hand, suits);
  const PublicPosition renamed =
      PublicPosition::Root(config, StreetKind::kFlop, renamed_board);

  const InfoSetKey key{HistoryId(7), position.observation(),
                       ObservePrivate(config, hand, position)};
  const InfoSetKey renamed_key{
      HistoryId(7), renamed.observation(),
      ObservePrivate(config, renamed_hand, renamed)};
  CHECK(key == renamed_key);
}

}  // namespace
}  // namespace poker
