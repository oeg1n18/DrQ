import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from src.SAC.state.buffer import ReplayBuffer
from src.SAC.state.networks import PolicyNetwork, QNetwork
from torch.utils.tensorboard import SummaryWriter

class SAC_Agent:
    def __init__(self, cfg):
        self.state_dim = cfg["state_dim"] # [cos(theta), sin(theta), theta_dot]
        self.action_dim = cfg["action_dim"]  # [torque] in[-2,2]
        self.lr_pi = cfg["policy_lr"]
        self.lr_q = cfg["critic_lr"]
        self.gamma = cfg["discount"]
        self.batch_size = cfg["batch_size"]
        self.buffer_limit = cfg["replay_size"]
        self.tau = cfg["soft_target_update"]
        self.init_alpha = cfg["init_alpha"]
        self.target_entropy = -self.action_dim  # == -1
        self.lr_alpha = cfg["lr_alpha"]
        self.DEVICE = cfg["device"]
        self.memory = ReplayBuffer(self.buffer_limit, self.state_dim, self.action_dim, self.DEVICE)
        self.chkpt_dir = cfg["chkpt_dir"]
        self.log_freq = cfg["log_freq"]
        self.train_counter = 0
        self.writer = SummaryWriter()

        self.log_alpha = torch.tensor(np.log(self.init_alpha)).to(self.DEVICE)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr_alpha)

        self.PI = PolicyNetwork(self.state_dim, self.action_dim, self.lr_pi).to(self.DEVICE)
        self.Q1 = QNetwork(self.state_dim, self.action_dim, self.lr_q).to(self.DEVICE)
        self.Q1_target = QNetwork(self.state_dim, self.action_dim, self.lr_q).to(self.DEVICE)
        self.Q2 = QNetwork(self.state_dim, self.action_dim, self.lr_q).to(self.DEVICE)
        self.Q2_target = QNetwork(self.state_dim, self.action_dim, self.lr_q).to(self.DEVICE)

        self.Q1_target.load_state_dict(self.Q1.state_dict())
        self.Q2_target.load_state_dict(self.Q2.state_dict())

    def choose_action(self, s):
        with torch.no_grad():
            action, log_prob = self.PI.sample(s.to(self.DEVICE))
        return action, log_prob

    def calc_target(self, mini_batch):
        s, a, r, s_prime, done = mini_batch
        with torch.no_grad():
            a_prime, log_prob_prime = self.PI.sample(s_prime)
            entropy = - self.log_alpha.exp() * log_prob_prime
            q1_target, q2_target = self.Q1_target(s_prime, a_prime), self.Q2_target(s_prime, a_prime)
            q_target = torch.min(q1_target, q2_target)
            target = r + self.gamma * done * (q_target + entropy)
        return target

    def learn(self):
        self.train_counter += 1
        o, a, r, o_, d = self.memory.sample(self.batch_size)
        td_target = self.calc_target((o, a, r, o_, d))

        #### Q1 train ####
        self.Q1.optimizer.zero_grad()
        q1_loss = F.mse_loss(self.Q1(o, a), td_target)
        q1_loss.mean().backward()
        # nn.utils.clip_grad_norm_(self.q1.parameters(), 1.0)
        self.Q1.optimizer.step()
        #### Q1 train ####


        #### Q2 train ####

        self.Q2.optimizer.zero_grad()
        q2_loss = F.mse_loss(self.Q2(o, a), td_target)
        q2_loss.mean().backward()
        # nn.utils.clip_grad_norm_(self.q2.parameters(), 1.0)
        self.Q2.optimizer.step()
        #### Q2 train ####

        #### pi train ####
        actions, log_prob = self.PI.sample(o)
        entropy = -self.log_alpha.exp() * log_prob
        q1, q2 = self.Q1(o, actions), self.Q2(o, actions)
        q = torch.min(q1, q2)
        pi_loss = -(q + entropy)  # for gradient ascent
        self.PI.optimizer.zero_grad()
        pi_loss.mean().backward()
        # nn.utils.clip_grad_norm_(self.pi.parameters(), 2.0)
        self.PI.optimizer.step()
        #### pi train ####

        #### alpha train ####
        self.log_alpha_optimizer.zero_grad()
        alpha_loss = -(self.log_alpha.exp() * (log_prob + self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
        #### alpha train ####

        #### Q1, Q2 soft-update ####
        for param_target, param in zip(self.Q1_target.parameters(), self.Q1.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)
        for param_target, param in zip(self.Q2_target.parameters(), self.Q2.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)
        #### Q1, Q2 soft-update ####

        if self.train_counter % self.log_freq == 0:
            self.writer.add_scalar("agent/Loss/q1_loss", q1_loss.detach().cpu().mean(), self.train_counter)
            self.writer.add_scalar("agent/Loss/q2_loss", q2_loss.detach().cpu().mean(), self.train_counter)
            self.writer.add_scalar("agent/Loss/pi_loss", pi_loss.detach().cpu().numpy().mean(), self.train_counter)
            self.writer.add_scalar("agent/Loss/alpha_loss", alpha_loss.detach().cpu().mean(), self.train_counter)
            self.writer.add_scalar("agent/alpha", self.log_alpha.exp().detach().cpu(), self.train_counter)

    def save_models(self):
        torch.save(self.PI.state_dict(), self.chkpt_dir + '/saved_models/sac/policy')
        torch.save(self.Q1.state_dict(), self.chkpt_dir + '/saved_models/sac/Q1')
        torch.save(self.Q2.state_dict(), self.chkpt_dir + '/saved_models/sac/Q2')
        torch.save(self.Q1_target.state_dict(), self.chkpt_dir + '/saved_models/sac/Q1_target')
        torch.save(self.Q2.state_dict(), self.chkpt_dir + '/saved_models/sac/Q2_target')
        self.memory.save(self.chkpt_dir + '/saved_replay_buffer/')

    def load_models(self):
        self.PI.load_state_dict(torch.load(self.chkpt_dir + '/saved_models/sac/policy'))
        self.Q1.load_state_dict(torch.load(self.chkpt_dir + '/saved_models/sac/Q1'))
        self.Q2.load_state_dict(torch.load(self.chkpt_dir + '/saved_models/sac/Q2'))
        self.Q1_target.load_state_dict(torch.load(self.chkpt_dir + '/saved_models/sac/Q1_target'))
        self.Q2_target.load_state_dict(torch.load(self.chkpt_dir + '/saved_models/sac/Q2_target'))
        self.memory.load(self.chkpt_dir + '/saved_replay_buffer/')
