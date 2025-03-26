import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np

device = torch.device("cpu")

class Linear_QNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, 512)
        self.linear2 = nn.Linear(512, 128)
        self.linear3 = nn.Linear(128, 32)
        self.linear4 = nn.Linear(32, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class CNN_QNet(nn.Module):
    def __init__(self, input_channels, output_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=7, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2)
        self.fc1 = nn.Linear(2*2*128, 512)
        self.output = nn.Linear(512, output_size)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.permute(2, 0, 1) # Change shape from (h, w, 3) to (3, h, w)
        else:
            x = x.permute(0, 3, 1, 2) # Change shape from (batch, h, w, 3) to (batch, 3, h, w)
        x = x / 255.0  # Normalize state space
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = x.reshape(-1, 2*2*128)  # Reshape for the fully connected layer
        x = F.relu(self.fc1(x))
        x = self.output(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model.to(device)
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float).to(device)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float).to(device)
        action = torch.tensor(np.array(action), dtype=torch.long).to(device)
        reward = torch.tensor(np.array(reward), dtype=torch.float).to(device)
        done = torch.tensor(np.array(done), dtype=torch.bool).to(device)

        if done.dim() == 0:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = done.unsqueeze(0)

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()

        # q_lr = 0.9
        # for idx in range(len(done)):
        #     # Q = target[idx][torch.argmax(action[idx]).item()]
        #     # Q_new = Q + q_lr*(reward[idx] - Q)
        #     Q_new = reward[idx]
        #     if not done[idx]:
        #         Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
        #         # Q_new = Q + q_lr*(reward[idx] + self.gamma * torch.max(self.model(next_state[idx])) - Q)

        #     target[idx][torch.argmax(action[idx]).item()] = Q_new

        # Parallelized Q update
        Q_new = reward.clone()
        not_done_indices = ~done

        max_next_Q_vals = torch.max(self.model(next_state), dim=1)[0]
        Q_new[not_done_indices] += self.gamma * max_next_Q_vals[not_done_indices]

        # Update Q-values in the target tensor
        for idx in range(done.dim()):
            target[idx][torch.argmax(action[idx]).item()] = Q_new[idx]
    
        
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()



