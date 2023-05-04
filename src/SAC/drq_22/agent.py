import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from buffer import ReplayBuffer
from networks import PolicyNetwork, QNetwork
from torch.utils.tensorboard import SummaryWriter
from src.SAC.autoencoder.AutoEncoder import AutoencoderModule
import torchvision
import pytorch_lightning as pl
import time

class SAC_Agent:
    def __init__(self, cfg):
        self.state_dim = cfg["state_dim"]  # [cos(theta), sin(theta), theta_dot]
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
        self.memory = ReplayBuffer(self.buffer_limit, (8, 84, 84), self.action_dim, self.DEVICE)
        self.chkpt_dir = cfg["chkpt_dir"]
        self.log_freq = cfg["log_freq"]
        self.train_counter = 0
        self.writer = SummaryWriter()

        self.log_alpha = torch.tensor(np.log(self.init_alpha)).to(self.DEVICE)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr_alpha)

        self.augmentation = torchvision.transforms.RandomCrop(size=(84, 84), padding=4, padding_mode='edge')

        ### ================= DrQ Averaging ==================
        self.K, self.M = 2, 2

        self.critic_encoder = AutoencoderModule.load_from_checkpoint(
            "/home/ollie/Documents/Github/DeepLearningDrQ/SAC/deepmind_pendulum_swingup/AutoEncoder/models/epoch=17-step=3528.ckpt",
            frame_size=(84, 84),
            latent_dim=50,
            frame_stack=8).to(self.DEVICE)

        self.policy_encoder = AutoencoderModule.load_from_checkpoint(
            "/home/ollie/Documents/Github/DeepLearningDrQ/SAC/deepmind_pendulum_swingup/AutoEncoder/models/epoch=17-step=3528.ckpt",
            frame_size=(84, 84),
            latent_dim=50,
            frame_stack=8).to(self.DEVICE)

        self.PI = PolicyNetwork(self.state_dim, self.action_dim, self.lr_pi).to(self.DEVICE)
        self.Q1 = QNetwork(self.state_dim, self.action_dim, self.lr_q).to(self.DEVICE)
        self.Q1_target = QNetwork(self.state_dim, self.action_dim, self.lr_q).to(self.DEVICE)
        self.Q2 = QNetwork(self.state_dim, self.action_dim, self.lr_q).to(self.DEVICE)
        self.Q2_target = QNetwork(self.state_dim, self.action_dim, self.lr_q).to(self.DEVICE)

        self.Q1_optimizer = optim.Adam(self.Q1.parameters(), lr=self.lr_q)
        self.Q2_optimizer = optim.Adam(self.Q2.parameters(), lr=self.lr_q)
        self.PI_optimizer = optim.Adam(self.PI.parameters(), lr=self.lr_pi)

        self.Q1_target.load_state_dict(self.Q1.state_dict())
        self.Q2_target.load_state_dict(self.Q2.state_dict())

    def choose_action(self, s):
        with torch.no_grad():
            s = self.policy_encoder.encode(s.to(self.DEVICE), detach_encoder=True)
            action, log_prob = self.PI.sample(s.to(self.DEVICE))
        return action, log_prob

    def calc_target(self, mini_batch):
        s, a, r, s_prime, done = mini_batch
        targets = []
        with torch.no_grad():
            for i in range(self.K):
                s_prime_aug = self.augmentation(s_prime)
                s_prime_aug = self.critic_encoder.encode(s_prime_aug, detach_encoder=True)
                a_prime_aug, log_prob_prime_aug = self.PI.sample(s_prime_aug)
                entropy = - self.log_alpha.exp() * log_prob_prime_aug
                q1_target, q2_target = self.Q1_target(s_prime_aug, a_prime_aug), self.Q2_target(s_prime_aug,
                                                                                                a_prime_aug)
                q_target = torch.min(q1_target, q2_target)
                targets.append(r + self.gamma * done * (q_target + entropy))
            if len(targets) >= 2:
                var = F.mse_loss(targets[0], targets[1])
                self.writer.add_scalar("drq/td_target_var", var, self.train_counter)
        return torch.mean(torch.stack(targets), dim=0)

    def learn(self):
        self.train_counter += 1
        o, a, r, o_, d = self.memory.sample(self.batch_size)
        td_target = self.calc_target((o, a, r, o_, d))

        #### Q1 train ####
        self.Q1_optimizer.zero_grad()
        q1_losses = []
        for i in range(self.M):
            o_aug = self.augmentation(o)
            q_o = self.critic_encoder.encode(o_aug, detach_encoder=False)
            q1_losses.append(F.mse_loss(self.Q1(q_o, a), td_target))
        q1_loss = torch.mean(torch.stack(q1_losses), dim=0)
        q1_loss.mean().backward()
        self.Q1_optimizer.step()
        #### Q1 train ####


        with torch.no_grad():
            if len(q1_losses) >= 2:
                var = F.mse_loss(q1_losses[0], q1_losses[1])
                self.writer.add_scalar("drq/Q1_loss_var", var, self.train_counter)

        #### Q2 train ####
        self.Q2_optimizer.zero_grad()
        q2_losses = []
        for i in range(self.M):
            o_aug = self.augmentation(o)
            q_o = self.critic_encoder.encode(o_aug)
            q2_losses.append(F.mse_loss(self.Q2(q_o, a), td_target))
        q2_loss = torch.mean(torch.stack(q2_losses), dim=0)
        q2_loss.mean().backward()
        self.Q2_optimizer.step()
        #### Q2 train ####

        with torch.no_grad():
            if len(q1_losses) >= 2:
                var = F.mse_loss(q1_losses[0], q1_losses[1])
                self.writer.add_scalar("drq/Q2_loss_var", var, self.train_counter)

        #### pi train ####
        pi_o = self.policy_encoder.encode(o, detach_encoder=True)
        actions, log_prob = self.PI.sample(pi_o)
        entropy = -self.log_alpha.exp() * log_prob
        q1, q2 = self.Q1(pi_o, actions), self.Q2(pi_o, actions)
        q = torch.min(q1, q2)
        pi_loss = -(q + entropy)  # for gradient ascent
        self.PI_optimizer.zero_grad()
        pi_loss.mean().backward()
        self.PI_optimizer.step()
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

        ### Keep Encoder Conv Weights the same but allow the critic to update them###
        """
        copy_keys = ['encoder.conv1.weight', 'encoder.conv1.bias', 'encoder.conv2.weight', 'encoder.conv2.bias',
                     'encoder.conv3.weight', 'encoder.conv3.bias', 'encoder.conv4.weight', 'encoder.conv4.bias']
        target_dict = self.policy_encoder.state_dict()
        for key, value in self.critic_encoder.state_dict().items():
            if key in copy_keys:
                target_dict[key] = value
        self.policy_encoder.load_state_dict(target_dict)
        """


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
