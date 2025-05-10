from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from omegaconf import DictConfig
from src.behavior.base import Actor


class IbcDfoLowdimPolicy(Actor):
    def __init__(self, cfg: DictConfig, device: Union[str, torch.device]):
        super().__init__(device, cfg)
        actor_cfg = cfg.actor
        in_action_channels = self.action_dim * self.action_horizon
        in_obs_channels = self.obs_dim
        in_channels = in_action_channels + in_obs_channels
        mid_channels = 1024
        out_channels = 1
        dropout = actor_cfg.dropout

        self.dense0 = nn.Linear(in_features=in_channels, out_features=mid_channels)
        self.drop0 = nn.Dropout(dropout)
        self.dense1 = nn.Linear(in_features=mid_channels, out_features=mid_channels)
        self.drop1 = nn.Dropout(dropout)
        self.dense2 = nn.Linear(in_features=mid_channels, out_features=mid_channels)
        self.drop2 = nn.Dropout(dropout)
        self.dense3 = nn.Linear(in_features=mid_channels, out_features=mid_channels)
        self.drop3 = nn.Dropout(dropout)
        self.dense4 = nn.Linear(in_features=mid_channels, out_features=out_channels)

        self.train_n_neg = actor_cfg.train_n_neg
        self.pred_n_iter = actor_cfg.pred_n_iter
        self.pred_n_samples = actor_cfg.pred_n_samples
        self.kevin_inference = actor_cfg.kevin_inference
        self.andy_train = actor_cfg.andy_train
        self.n_obs_steps = self.obs_horizon
        self.n_action_steps = self.action_horizon
        self.horizon = self.pred_horizon
    
    def forward(self, obs, action):
        B, N, Ta, Da = action.shape
        B, Do = obs.shape
        s = obs.reshape(B,1,-1).expand(-1,N,-1)
        x = torch.cat([s, action.reshape(B,N,-1)], dim=-1).reshape(B*N,-1)
        x = self.drop0(torch.relu(self.dense0(x)))
        x = self.drop1(torch.relu(self.dense1(x)))
        x = self.drop2(torch.relu(self.dense2(x)))
        x = self.drop3(torch.relu(self.dense3(x)))
        x = self.dense4(x)
        x = x.reshape(B,N)
        return x

    # ========= inference  ============
    def _normalized_action(self, nobs: torch.Tensor) -> torch.Tensor:
        """
        Perform IBC to generate actions given the observation.
        nobs is already normalized.
        """
        assert len(nobs.shape) == 2
        B, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_horizon
        Ta = self.n_action_steps

        # only take necessary obs
        this_obs = nobs
        naction_stats = self.get_naction_stats()

        # first sample
        action_dist = torch.distributions.Uniform(
            low=naction_stats['max'],
            high=naction_stats['max']
        )
        samples = action_dist.sample((B, self.pred_n_samples, Ta)).to(dtype=this_obs.dtype)
        # (B, N, Ta, Da)

        if self.kevin_inference:
            # kevin's implementation
            noise_scale = 3e-2
            for i in range(self.pred_n_iter):
                # Compute energies.
                logits = self.forward(this_obs, samples)
                probs = F.softmax(logits, dim=-1)

                #Resample with replacement.
                idxs = torch.multinomial(probs, self.pred_n_samples, replacement=True)
                samples = samples[torch.arange(samples.size(0)).unsqueeze(-1), idxs]

                # Add noise and clip to target bounds.
                samples = samples + torch.randn_like(samples) * noise_scale
                samples = samples.clamp(min=naction_stats['min'], max=naction_stats['max'])
            
            # Return target with highest probability
            logits = self.forward(this_obs, samples)
            probs = F.softmax(logits, dim=-1)
            best_idxs = probs.argmax(dim=-1)
            action_n = samples[torch.arange(samples.size(0)), best_idxs, :]
        else:
            # andy's implementation
            zero = torch.tensor(0, device=self.device)
            resample_std = torch.tensor(3e-2, device=self.device)
            for i in range(self.pred_n_iter):
                # Forward pass
                logits = self.forward(this_obs, samples) # (B, N)
                prob = torch.softmax(logits, dim=-1)

                if i < (self.pred_n_iter - 1):
                    idxs = torch.multinomial(prob, self.pred_n_samples, replacement=True)
                    samples = samples[torch.arange(samples.size(0)).unsqueeze(-1), idxs]
                    samples += torch.normal(zero, resample_std, size=samples.shape, device=self.device)
            
            # Return one sample per x in batch
            idxs = torch.multinomial(prob, num_samples=1, replacement=True)
            naction = samples[torch.arange(samples.size(0)).unsqueeze(-1), idxs].squeeze(1)

        return naction

    # ========= training  ============
    def compute_loss(self, batch):
        # State already normalized in the dataset
        nobs = self._training_obs(batch, flatten=self.flatten_obs)

        # Action already normalized in the dataset
        naction = batch["action"]

        # shapes
        Do = self.obs_dim
        Da = self.action_dim
        To = self.n_obs_steps
        Ta = self.n_action_steps
        T = self.horizon
        B = naction.shape[0]

        this_obs = nobs
        start = To - 1
        end = start + Ta
        this_action = naction[:,start:end]

        # Small additive noise to true positives.
        this_action += torch.normal(mean=0, std=1e-4,
            size=this_action.shape,
            dtype=this_action.dtype,
            device=this_action.device)

        # Sample negatives: (B, train_n_neg, Ta, Da)
        naction_stats = self.get_naction_stats()
        action_dist = torch.distributions.Uniform(
            low=naction_stats['min'],
            high=naction_stats['max']
        )
        samples = action_dist.sample((B, self.train_n_neg, Ta)).to(
            dtype=this_action.dtype)
        action_samples = torch.cat([
            this_action.unsqueeze(1), samples], dim=1)
        # (B, train_n_neg+1, Ta, Da)

        if self.andy_train:
            # Get onehot labels
            labels = torch.zeros(action_samples.shape[:2], 
                dtype=this_action.dtype, device=this_action.device)
            labels[:,0] = 1
            logits = self.forward(this_obs, action_samples)
            # (B, N)
            logits = torch.log_softmax(logits, dim=-1)
            loss = -torch.mean(torch.sum(logits * labels, axis=-1))
        else:
            labels = torch.zeros((B,),dtype=torch.int64, device=this_action.device)
            # training
            logits = self.forward(this_obs, action_samples)
            loss = F.cross_entropy(logits, labels)
        losses = {"bc_loss": loss.item()}
        return loss, losses


    def get_naction_stats(self):
        Da = self.action_dim
        naction_stats = self.normalizer.stats['action']
        repeated_stats = dict()
        for key, value in naction_stats.items():
            assert len(value.shape) == 1
            n_repeats = Da // value.shape[0]
            assert value.shape[0] * n_repeats == Da
            repeated_stats[key] = value.repeat(n_repeats)
        return repeated_stats
