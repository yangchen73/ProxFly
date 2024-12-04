import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

def check_validity(tensor, name):
    if not torch.isfinite(tensor).all():
        raise ValueError(f"The parameter {name} has invalid values: {tensor}")


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Tanh):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input: vector x, [x0, x1, x2]
    output: [x0 + discount * x1 + discount^2 * x2,  x1 + discount * x2, x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):
    
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):
    def __init__(self, obs_dim, act_dim, latent_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([latent_dim + obs_dim + act_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, z_obs_act):
        mu = self.mu_net(z_obs_act)
        std = torch.exp(self.log_std)
        check_validity(mu, "mu")
        check_validity(std, "std")
        return Normal(mu, std)
    
    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)
class MLPCritic(nn.Module):

    def __init__(self, latent_dim, hidden_sizes, activation, output_activation=nn.LeakyReLU):
        super().__init__()
        self.output_activation = output_activation
        self.v_net = mlp([latent_dim] + list(hidden_sizes) + [1], activation, output_activation=self.output_activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.

class VAE(nn.Module):
    def __init__(self, obs_act_dim, latent_dim, hidden_sizes, activation=nn.ReLU):
        super().__init__()
        # input shape [400, 24], output shap [1, 393]
        self.conv1d = nn.Conv1d(in_channels=obs_act_dim, out_channels=1, kernel_size=8, stride=1)
        # MLP after convolution to produce latent dimension
        conv_out_size = (400 - 8 + 1)
        self.encoder = mlp([conv_out_size] + list(hidden_sizes) + [latent_dim * 2], activation)
        #self.decoder = mlp([latent_dim] + list(hidden_sizes[::-1]) + [obs_act_dim * 400], activation)

    def encode(self, obs_act_history):
        obs_act_history = obs_act_history.unsqueeze(0)
        conv_out = self.conv1d(obs_act_history.permute(0, 2, 1))  
        conv_out_flat = conv_out.view(conv_out.size(0), -1)  
        z_params = self.encoder(conv_out_flat)
        mu, log_std = z_params.chunk(2, dim=-1)
        std = torch.exp(log_std)
        return mu, std

    def reparameterize(self, mu, std):
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, obs_act_history):
        mu, std = self.encode(obs_act_history)
        z = self.reparameterize(mu, std)
        return z, mu, std
    
class MultiAgentProcessor(nn.Module):
    def __init__(self, num_agents, input_size, hidden_sizes, output_size1 = 128, output_size2 = 16):
        super(MultiAgentProcessor, self).__init__()
        self.num_agents = num_agents
        self.mlp1 = mlp([input_size] + hidden_sizes + [output_size1], activation=nn.ReLU)
        self.mlp2 = mlp([output_size1] + hidden_sizes + [output_size2], activation=nn.ReLU)

    def process_tensor(self, tensor):
        agent_outputs = []
        for i in range(self.num_agents):
            input_tensor = tensor[:, i] # torch.Size([20])
            input_tensor = input_tensor.unsqueeze(0)  # torch.Size([1, 20])
            agent_output = self.mlp1(input_tensor)  # torch.Size([1, 128])
            agent_outputs.append(agent_output.squeeze(0))  

        agent_outputs = torch.stack(agent_outputs, dim=0)
        pooled_output = torch.max(agent_outputs, dim=0).values
        output = self.mlp2(pooled_output.unsqueeze(0)).squeeze(0)

        return output
    
class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, latent_dim=32,
                 hidden_sizes=(64, 64), activation=nn.LeakyReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        obs_act_his_dim = obs_dim + act_dim
        self.vae = VAE(obs_act_his_dim, latent_dim, hidden_sizes, activation)
        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], latent_dim, hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(latent_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.v = MLPCritic(latent_dim, hidden_sizes, activation)
        self.N = 10
        self.multiagent_processor =  MultiAgentProcessor(input_size=20, num_agents=10, hidden_sizes=hidden_sizes, output_size1=128, output_size2=16)

    def step(self, obs_his, act_his):
        tensor = torch.zeros(20, self.N)
        with torch.no_grad():  
            output = self.multiagent_processor.process_tensor(tensor)
            act_recent = act_his[-1]
            obs_recent = obs_his[-1]
            obs_act_his = torch.cat((obs_his, act_his), dim=-1) 
            z, _, _ = self.vae(obs_act_his)
            combine = torch.cat((z, obs_recent.unsqueeze(0), act_recent.unsqueeze(0)), dim=-1)
            pi = self.pi._distribution(combine)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(z) 

        return a[0].numpy(), v[0].numpy(), logp_a[0].numpy()
    
    def act(self, obs):
        return self.step(obs)[0]