#include <iostream>
#include <memory>
#include "src/poker.pb.h"
#include "src/hand_evaluator.h"
#include "src/cfr_solver.h"
#include "src/game_tree.h"
#include "src/strategy.h"
#include "src/info_set.h"
#include "src/hand_range.h"
#include "src/utility_calculator.h"

class PokerSimulator {
public:
    void simulate() {}
    double payoff_for_action(poker::BoardState state) {
        // Use hand evaluator to determine payoffs
        poker::HandEvaluator evaluator;
        
        // Create example hole cards for two players
        poker::Hand player1_hand;
        poker::Card* card1 = player1_hand.add_cards();
        card1->set_rank(14); // Ace
        card1->set_suit(poker::Suit::SPADES);
        poker::Card* card2 = player1_hand.add_cards();
        card2->set_rank(13); // King
        card2->set_suit(poker::Suit::SPADES);
        
        poker::Hand player2_hand;
        poker::Card* card3 = player2_hand.add_cards();
        card3->set_rank(10); // Ten
        card3->set_suit(poker::Suit::HEARTS);
        poker::Card* card4 = player2_hand.add_cards();
        card4->set_rank(10); // Ten
        card4->set_suit(poker::Suit::DIAMONDS);
        
        try {
            // Compare hands and determine winner
            int result = evaluator.compare_hands(player1_hand, player2_hand, state);
            return result; // 1 if player1 wins, -1 if player2 wins, 0 if tie
        } catch (const std::exception& e) {
            // Handle error
            std::cerr << "Error evaluating hands: " << e.what() << std::endl;
        }
        
        return 0.0;
    }
};

int main() {
    // Create poker configuration
    poker::PokerConfig config;
    config.add_bet_sizes(0.25);
    config.add_bet_sizes(0.5);
    config.add_bet_sizes(1.0);
    config.set_starting_stack_size(100);
    config.set_small_blind(1);
    config.set_big_blind(2);

    // Create CFR solver with the configuration
    poker::CFRSolver solver(config);
    std::cout << "Poker Solver Initialized" << std::endl;

    // Run CFR iterations
    std::cout << "Running CFR algorithm..." << std::endl;
    solver.run(10); // Reduced from 1000 to 10 iterations for faster execution
    
    // Get the equilibrium strategy
    poker::Strategy equilibrium = solver.get_equilibrium_strategy();
    
    // Print some information about the strategy
    std::cout << "CFR algorithm completed" << std::endl;
    std::cout << "Number of information sets: " << equilibrium.get_info_sets().size() << std::endl;
    
    // Save the computed strategy
    solver.save_strategy("equilibrium_strategy.txt");
    std::cout << "Strategy saved to equilibrium_strategy.txt" << std::endl;
    
    // Create a board state with some cards
    poker::BoardState state;
    state.set_stack_a(100);
    state.set_stack_b(100);
    state.set_pot(20);
    state.set_folded_player(-1); // No player has folded
    state.set_street(poker::Street::RIVER);
    state.set_all_in(false);
    state.set_player_to_act(0);
    
    // Add player contributions to the pot
    state.add_player_contribution(10); // Player 0 contribution
    state.add_player_contribution(10); // Player 1 contribution
    
    // Add 5 community cards (flop, turn, river)
    for (int i = 2; i <= 6; i++) {
        poker::Card* card = state.add_cards();
        card->set_rank(i);
        card->set_suit(poker::Suit::HEARTS);
    }
    
    // Test hand evaluator with a specific hand
    poker::HandEvaluator evaluator;
    poker::Hand hand;
    
    // Create a flush (all hearts)
    for (int i = 10; i <= 14; i++) {
        poker::Card* card = hand.add_cards();
        card->set_rank(i);
        card->set_suit(poker::Suit::HEARTS);
    }
    
    try {
        // Evaluate a specific 5-card hand
        poker::HandEvaluation eval = evaluator.evaluate(hand);
        std::cout << "Hand evaluation: " << static_cast<int>(eval.rank) << std::endl;
        
        // Create player hole cards
        poker::Hand player1_hand;
        poker::Card* p1c1 = player1_hand.add_cards();
        p1c1->set_rank(14); // Ace
        p1c1->set_suit(poker::Suit::SPADES);
        poker::Card* p1c2 = player1_hand.add_cards();
        p1c2->set_rank(13); // King
        p1c2->set_suit(poker::Suit::SPADES);
        
        poker::Hand player2_hand;
        poker::Card* p2c1 = player2_hand.add_cards();
        p2c1->set_rank(10); // Ten
        p2c1->set_suit(poker::Suit::HEARTS);
        poker::Card* p2c2 = player2_hand.add_cards();
        p2c2->set_rank(10); // Ten
        p2c2->set_suit(poker::Suit::DIAMONDS);
        
        // Compare hands
        int result = evaluator.compare_hands(player1_hand, player2_hand, state);
        std::cout << "Hand comparison: " << result << std::endl;
        
        // Test hand range functionality
        poker::HandRange range;
        range.set_from_string("AA,KK,QQ,JJ,AKs");
        std::cout << "Hand range created with " << range.get_all_hands().size() << " hands" << std::endl;
        
        // Test utility calculator
        poker::UtilityCalculator utility_calc;
        double equity = utility_calc.calculate_equity(player1_hand, range, state);
        std::cout << "Equity of AKs against range: " << equity << std::endl;
        
        // Test expected value calculation
        double ev = utility_calc.calculate_expected_value(state, player1_hand, range, 0);
        std::cout << "Expected value: " << ev << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    
    return 0;
}
