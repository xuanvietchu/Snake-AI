import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point, BLOCK_SIZE, WINDOW_W, WINDOW_H
from model import Linear_QNet, QTrainer
from helper import plot
import time

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
# LR = 0.03

class Agent:

    def __init__(self, test = False):
        self.n_games = 0
        self.epsilon = 0.05 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(17, 3)
        self.trainer = None
        if not test:
            self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        head = game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        # Calculate distances to walls (or boundaries)
        distance_to_left_wall = head.x
        distance_to_right_wall = WINDOW_W - head.x
        distance_to_top_wall = head.y
        distance_to_bottom_wall = WINDOW_H - head.y

        # Calculate distances to food (x and y direction)
        distance_to_food_x = game.food.x - head.x
        distance_to_food_y = game.food.y - head.y

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y,  # food down

            distance_to_left_wall / WINDOW_W,
            distance_to_right_wall / WINDOW_W, 
            distance_to_top_wall / WINDOW_H,
            distance_to_bottom_wall / WINDOW_H,
            
            distance_to_food_x / WINDOW_W,
            distance_to_food_y / WINDOW_H,
            ]

        return np.array(state, dtype=int)


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 100 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 400) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    
    while True:
        start_time = time.time()  # Record start time for the game
        
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset(agent.n_games)
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                try:
                    torch.save(agent.model.state_dict(), './model/best.pth') # occasionally fail
                except:
                    print(f"fail to save best model at game {agent.n_games}")
            elif agent.n_games % 10 == 0:
                try:
                    torch.save(agent.model.state_dict(), './model/last.pth') # occasionally fail
                except:
                    print(f"fail to save last model at game {agent.n_games}")
                

            print(LR, agent.n_games)
            
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)

            # if agent.n_games == 100: # Early stop if performance not satisfied
            #     if mean_score < 0.1 or record < 4:
            #         elapsed_time = time.time() - start_time  # Calculate elapsed time
            #         print(f'Game {agent.n_games}, Score: {sum(plot_scores[-30:])/30}, Record: {record}, Time taken: {elapsed_time*100:.2f} seconds')
            #         plot_file_name = f'LR_{lr}.png'
            #         plot(plot_scores, plot_mean_scores, agent.n_games, file_name=plot_file_name)
            #         break

            if agent.n_games == 200:
                if mean_score < 7:
                        print(f"Rerunning training due to mean_score {mean_score} < 7 (bad start)")
                        return False  # Signal to rerun training

            if agent.n_games == 1200:
                elapsed_time = time.time() - start_time  # Calculate elapsed time
                print(f'Game {agent.n_games}, Score: {sum(plot_scores[-30:])/10}, Record: {record}, Time taken: {elapsed_time*200:.2f} seconds')
                plot_file_name = f'LR_{lr}.png'
                plot(plot_scores, plot_mean_scores, agent.n_games, file_name=plot_file_name)
                return True


if __name__ == '__main__':
    # lr_values = [
    #     1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5,
    #     1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4,
    #     1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3, 9e-3,
    #     ]
    # # lr_values = [1]
    # for lr in lr_values:
    #     LR = lr
    #     train()
    LR = 6e-05
    lr = LR
    rerun = train()
    while not rerun:
        rerun = train()