import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Memory:
    def __init__(self):
        self.actions = []
        self.ee_pos = []
        self.logprobs = []
        self.rewards = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.ee_pos[:]
        del self.logprobs[:]
        del self.rewards[:]

class ActorCritic(nn.Module):
    def __init__(self, args, state_dim, action_dim, action_std):
        super(ActorCritic, self).__init__()
        self.actor_mean =  nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, action_dim),
                nn.Tanh()
                )
        self.actor_logstd =  nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, action_dim),
                )
        # critic
        self.critic = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, 1)
                )
    
        self.action_var = torch.full((action_dim,), action_std*action_std)
        
    def forward(self):
        raise NotImplementedError
    
    def act(self, state, memory):
        state = torch.unsqueeze(torch.from_numpy(state), 0).cuda().float()
        with torch.no_grad():
            action_mean = self.actor_mean(state).cpu()
            action_logstd = self.actor_logstd(state).cpu()
        
        d = action_mean.shape[-1]
        sample_size = action_mean.size()
        sample_std_normal = torch.normal(torch.zeros(sample_size), torch.ones(sample_size))
        action = action_mean + torch.exp(action_logstd) * sample_std_normal
        action_logprob = (2 * np.pi)**(-d/2) * torch.exp(- torch.norm(sample_std_normal))
        
        memory.ee_pos.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)
        
        return action.detach()
    
    def compute_logprobs(self, action_mean, action_var, action):
        k = action.size(-1)
        delta = action - action_mean
        result = -k/2. * torch.log(torch.Tensor([2 * np.pi]).cuda()) - \
            0.5 * torch.sum(torch.log(action_var), dim=-1) - \
            0.5 * torch.sum(delta**2 / action_var, dim=-1)
        return result

    def evaluate(self, state, action):
        action_mean = torch.squeeze(self.actor_mean(state))
        action_logstd = torch.squeeze(self.actor_logstd(state))
        action_std = torch.exp(action_logstd)
        action_var = action_std * action_std
        
        action_logprobs = self.compute_logprobs(action_mean, action_var, action)
        state_value = self.critic(state)

        return action_logprobs, torch.squeeze(state_value)

class PPO:
    def __init__(self, args, state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = ActorCritic(args, state_dim, action_dim, action_std).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(args, state_dim, action_dim, action_std).to(device)
        
        self.MseLoss = nn.MSELoss()
    
    def select_action(self, state, memory):
        return self.policy_old.act(state, memory).cpu().data.numpy().flatten()
   
    def update(self, memory):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward in reversed(memory.rewards):
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        old_ee_pos = torch.squeeze(torch.stack(memory.ee_pos)).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions)).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs)).detach()
        

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # shuffle data
            randperm = np.random.permutation(len(old_actions))
            old_ee_pos = old_ee_pos[randperm]
            old_actions = old_actions[randperm]
            old_logprobs = old_logprobs[randperm]
            rewards = rewards[randperm]

            batch_start = 0
            batch_size = 32
            while batch_start < len(old_actions):
                # images_batch = old_images[batch_start: batch_start+batch_size].cuda()
                ee_pos_batch = old_ee_pos[batch_start: batch_start+batch_size].cuda()
                actions_batch = old_actions[batch_start: batch_start+batch_size].cuda()
                logprobs_batch = old_logprobs[batch_start: batch_start+batch_size].cuda()
                rewards_batch = rewards[batch_start: batch_start+batch_size].cuda()

                # Evaluating old actions and values :
                logprobs, state_values = self.policy.evaluate(ee_pos_batch, actions_batch)
                
                # Finding the ratio (pi_theta / pi_theta__old):
                ratios = torch.exp(logprobs - logprobs_batch.detach())

                # Finding Surrogate Loss:
                advantages = rewards_batch - state_values.detach()   
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
                loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values.float(), rewards_batch.float())
                
                # take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
            
                batch_start += batch_size

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        


