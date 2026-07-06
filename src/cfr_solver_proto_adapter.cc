#include "src/cfr_solver_proto_adapter.h"

#include <cmath>

namespace poker {
namespace {

SuitKind ToSuitKind(Suit suit) {
  switch (suit) {
    case Suit::DIAMONDS:
      return SuitKind::kDiamonds;
    case Suit::CLUBS:
      return SuitKind::kClubs;
    case Suit::SPADES:
      return SuitKind::kSpades;
    case Suit::HEARTS:
    default:
      return SuitKind::kHearts;
  }
}

StreetKind ToStreetKind(Street street) {
  switch (street) {
    case Street::FLOP:
      return StreetKind::kFlop;
    case Street::TURN:
      return StreetKind::kTurn;
    case Street::RIVER:
      return StreetKind::kRiver;
    case Street::PREFLOP:
    default:
      return StreetKind::kPreflop;
  }
}

ActionKind ToActionKind(ActionType action_type) {
  switch (action_type) {
    case ActionType::BET:
      return ActionKind::kBet;
    case ActionType::FOLD:
      return ActionKind::kFold;
    case ActionType::CALL:
      return ActionKind::kCall;
    case ActionType::RAISE:
      return ActionKind::kRaise;
    case ActionType::CHECK:
      return ActionKind::kCheck;
    case ActionType::ALL_IN:
      return ActionKind::kAllIn;
    case ActionType::NO_ACTION:
    default:
      return ActionKind::kNoAction;
  }
}

}  // namespace

CardId CardIdFromProto(const Card& card) {
  return MakeCardId(card.rank(), ToSuitKind(card.suit()));
}

ComboId ComboIdFromProtoHand(const Hand& hand) {
  if (hand.cards_size() != 2) {
    return 0;
  }
  return CardsToComboId(CardIdFromProto(hand.cards(0)),
                        CardIdFromProto(hand.cards(1)));
}

GameAction GameActionFromProto(const Action& action) {
  return {ToActionKind(action.action()),
          static_cast<int>(std::lround(action.amount())),
          action.player()};
}

SolverConfig SolverConfigFromProto(const PokerConfig& config) {
  SolverConfig native;
  native.bet_sizes.assign(config.bet_sizes().begin(), config.bet_sizes().end());
  native.starting_stack_size = config.starting_stack_size();
  native.max_depth = config.max_depth();
  native.enable_logging = config.enable_logging();
  native.small_blind = config.small_blind();
  native.big_blind = config.big_blind();
  native.chance_samples = config.chance_samples();
  native.preflop_bet_sizes.assign(config.preflop_bet_sizes().begin(),
                                  config.preflop_bet_sizes().end());
  native.flop_bet_sizes.assign(config.flop_bet_sizes().begin(),
                               config.flop_bet_sizes().end());
  native.turn_bet_sizes.assign(config.turn_bet_sizes().begin(),
                               config.turn_bet_sizes().end());
  native.river_bet_sizes.assign(config.river_bet_sizes().begin(),
                                config.river_bet_sizes().end());
  native.regret_only_training = config.regret_only_training();
  native.max_info_sets = static_cast<int>(config.max_info_sets());
  native.max_public_states = static_cast<int>(config.max_public_states());
  native.num_training_threads = config.num_training_threads();
  native.warmup_iterations = config.warmup_iterations();
  return native;
}

PokerConfig SolverConfigToProto(const SolverConfig& config) {
  PokerConfig proto;
  for (double bet_size : config.bet_sizes) {
    proto.add_bet_sizes(bet_size);
  }
  proto.set_starting_stack_size(config.starting_stack_size);
  proto.set_max_depth(config.max_depth);
  proto.set_enable_logging(config.enable_logging);
  proto.set_small_blind(config.small_blind);
  proto.set_big_blind(config.big_blind);
  proto.set_chance_samples(config.chance_samples);
  for (double bet_size : config.preflop_bet_sizes) {
    proto.add_preflop_bet_sizes(bet_size);
  }
  for (double bet_size : config.flop_bet_sizes) {
    proto.add_flop_bet_sizes(bet_size);
  }
  for (double bet_size : config.turn_bet_sizes) {
    proto.add_turn_bet_sizes(bet_size);
  }
  for (double bet_size : config.river_bet_sizes) {
    proto.add_river_bet_sizes(bet_size);
  }
  proto.set_regret_only_training(config.regret_only_training);
  proto.set_max_info_sets(config.max_info_sets);
  proto.set_max_public_states(config.max_public_states);
  proto.set_num_training_threads(config.num_training_threads);
  proto.set_warmup_iterations(config.warmup_iterations);
  return proto;
}

GameState GameStateFromProto(const BoardState& state) {
  GameState native;
  native.stack[0] = state.stack_a();
  native.stack[1] = state.stack_b();
  native.pot = state.pot();
  native.street = ToStreetKind(state.street());
  native.all_in = state.all_in();
  native.folded_player = state.folded_player();
  native.player_to_act = state.player_to_act();
  native.player_contribution[0] =
      state.player_contribution_size() > 0
          ? static_cast<int>(std::lround(state.player_contribution(0)))
          : 0;
  native.player_contribution[1] =
      state.player_contribution_size() > 1
          ? static_cast<int>(std::lround(state.player_contribution(1)))
          : 0;
  for (const Card& card : state.cards()) {
    AddBoardCard(native, CardIdFromProto(card));
  }
  for (const Action& action : state.history().actions()) {
    native.history.push_back(GameActionFromProto(action));
  }
  return native;
}

}  // namespace poker
