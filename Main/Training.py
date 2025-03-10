from Classes.AI import *

if __name__ == '__main__':
    # level 2 - qlearning
    agent = AI(level=2, player=2)

    # Train by selfplaying
    print("Training Q-learning agent via self-play...")
    agent.self_play_training(episodes=40000, alpha=0.5,gamma=0.90, epsilon=0.5)

    # Save the learned Q-table
    agent.save_Q("qtable.pkl")
