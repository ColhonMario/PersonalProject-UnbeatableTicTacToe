from Classes.AI import *

if __name__ == '__main__':
    # level 2 - qlearning
    agent = AI(level=2, player=2)

    # Train by selfplaying
    print("Training Q-learning agent via self-play...")
    agent.training(episodes=80000, alpha=0.01, gamma=0.90, epsilon_start=0.5, epsilon_end=0.1)

    # Save the learned Q-table
    agent.save_Q("qtable.pkl")
