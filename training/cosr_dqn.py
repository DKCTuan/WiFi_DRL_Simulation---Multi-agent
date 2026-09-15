"""Structured, masked Double-DQN for centralized Co-SR scheduling.

The previous flat DQN gave every AP--STA group an unrelated output neuron.
That formulation cannot generalise across the roughly 1,560 candidate groups.
This version uses a shared scorer: Q(s, group_features).  Consequently it
learns what makes a group urgent/strong, rather than memorising an action ID.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

import config
from env.cosr_baselines import select_group_oldest
from env.cosr_env import CoSREnv


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_argmax(values: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    if values.shape != masks.shape:
        raise ValueError("values and masks must have identical shape")
    if not torch.all(masks.any(dim=1)):
        raise ValueError("every state must have at least one valid action")
    return values.masked_fill(~masks, -torch.inf).argmax(dim=1)


def imitation_scale_for_episode(*, episode: int, episodes: int,
                                imitation_weight: float, imitation_floor: float) -> float:
    """Decay BC guidance over training while retaining an optional safety anchor."""
    if imitation_weight < 0 or imitation_floor < 0:
        raise ValueError("imitation weight and floor must be non-negative")
    if imitation_floor > imitation_weight:
        raise ValueError("imitation_floor cannot exceed imitation_weight")
    decay = imitation_weight * max(0.0, 1.0 - episode / max(episodes * 0.75, 1))
    return max(imitation_floor, decay)


class GroupScoringNetwork(nn.Module):
    """Score all candidate groups from shared per-group, not action-ID, features."""

    def __init__(self, state_dim: int, groups: list[tuple]):
        super().__init__()
        action_dim = len(groups) + 1
        sta_ids = np.full((action_dim, 2), -1, dtype=np.int64)
        ap_ids = np.full((action_dim, 2), -1, dtype=np.int64)
        for action, group in enumerate(groups, start=1):
            for member, (ap_id, sta_id) in enumerate(group):
                sta_ids[action, member] = sta_id
                ap_ids[action, member] = ap_id
        self.register_buffer("group_sta_ids", torch.as_tensor(sta_ids))
        self.register_buffer("group_ap_ids", torch.as_tensor(ap_ids))
        self.register_buffer("member_mask", self.group_sta_ids >= 0)
        self.num_stas = state_dim // 3
        self.action_dim = action_dim
        # sum queue, max queue, max HoL, mean/max direct SINR, group size,
        # and the two normalised AP identifiers are enough to share scheduling
        # behaviour among arbitrary AP--STA pairs.
        self.group_encoder = nn.Sequential(nn.Linear(8, 96), nn.ReLU(), nn.Linear(96, 128), nn.ReLU())
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, 192), nn.ReLU(), nn.Linear(192, 128), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))

    def group_features(self, state: torch.Tensor) -> torch.Tensor:
        """Construct B x actions x 8 features without enumerating action IDs."""
        batch = state.shape[0]
        per_sta = state.reshape(batch, self.num_stas, 3)
        ids = self.group_sta_ids.clamp_min(0)
        values = per_sta[:, ids]  # B x actions x two members x [q, hol, sinr]
        members = self.member_mask.unsqueeze(0).expand(batch, -1, -1)
        queues = values[..., 0]
        hol = values[..., 1]
        sinr = values[..., 2]
        count = members.sum(dim=2).clamp_min(1)
        neg_inf = torch.full_like(queues, -torch.inf)
        max_queue = torch.where(members, queues, neg_inf).max(dim=2).values
        max_hol = torch.where(members, hol, neg_inf).max(dim=2).values
        max_sinr = torch.where(members, sinr, neg_inf).max(dim=2).values
        sum_queue = torch.where(members, queues, torch.zeros_like(queues)).sum(dim=2)
        mean_sinr = torch.where(members, sinr, torch.zeros_like(sinr)).sum(dim=2) / count
        group_size = members.sum(dim=2).to(state.dtype) / 2.0
        ap = self.group_ap_ids.to(state.dtype).unsqueeze(0).expand(batch, -1, -1)
        ap_norm = torch.where(members, ap / max(config.NUM_APS - 1, 1), torch.zeros_like(ap))
        min_ap = torch.where(members, ap_norm, torch.full_like(ap_norm, torch.inf)).min(dim=2).values
        max_ap = torch.where(members, ap_norm, neg_inf).max(dim=2).values
        # Idle's features are exactly zero; it remains a valid safety action.
        idle = ~members.any(dim=2)
        features = torch.stack((sum_queue, max_queue, max_hol, mean_sinr, max_sinr,
                                group_size, min_ap, max_ap), dim=-1)
        return torch.where(idle.unsqueeze(-1), torch.zeros_like(features), features)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        global_embedding = self.state_encoder(state).unsqueeze(1).expand(-1, self.action_dim, -1)
        group_embedding = self.group_encoder(self.group_features(state))
        return self.head(torch.cat((global_embedding, group_embedding), dim=-1)).squeeze(-1)


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    next_mask: np.ndarray


@dataclass
class ExpertExample:
    state: np.ndarray
    mask: np.ndarray
    action: int


class MaskedReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity, self.data, self.position = int(capacity), [], 0

    def __len__(self) -> int:
        return len(self.data)

    def add(self, item: Transition) -> None:
        if len(self.data) < self.capacity:
            self.data.append(item)
        else:
            self.data[self.position] = item
        self.position = (self.position + 1) % self.capacity

    def sample(self, size: int, rng: random.Random) -> list[Transition]:
        return rng.sample(self.data, size)


def _greedy_action(network: GroupScoringNetwork, state: np.ndarray, mask: np.ndarray,
                   device: torch.device) -> int:
    with torch.no_grad():
        values = network(torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0))
        valid = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
        return int(masked_argmax(values, valid).item())


def collect_expert_examples(env: CoSREnv, *, steps: int, seed: int) -> list[ExpertExample]:
    state, info = env.reset(seed=seed)
    examples: list[ExpertExample] = []
    for index in range(steps):
        mask = np.asarray(info["action_mask"], dtype=bool)
        action = select_group_oldest(env, mask)
        examples.append(ExpertExample(state.copy(), mask.copy(), int(action)))
        state, _, _, truncated, info = env.step(action)
        if truncated and index + 1 < steps:
            state, info = env.reset(seed=seed + index + 1)
    return examples


def expert_update(network: GroupScoringNetwork, optimizer: torch.optim.Optimizer,
                  examples: list[ExpertExample], batch_size: int, rng: random.Random,
                  device: torch.device, weight: float = 1.0) -> float:
    batch = rng.sample(examples, min(batch_size, len(examples)))
    states = torch.as_tensor(np.stack([x.state for x in batch]), dtype=torch.float32, device=device)
    masks = torch.as_tensor(np.stack([x.mask for x in batch]), dtype=torch.bool, device=device)
    actions = torch.as_tensor([x.action for x in batch], dtype=torch.long, device=device)
    loss = weight * nn.functional.cross_entropy(network(states).masked_fill(~masks, -torch.inf), actions)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(network.parameters(), 10.0)
    optimizer.step()
    return float(loss.item())


def evaluate_cosr_dqn(network: GroupScoringNetwork, *, episodes: int, warmup_steps: int,
                      steps: int, seed: int, arrival_rate: float, fixed_topology: bool,
                      scenario_seed: int = config.TRAIN_SCENARIO_SEED,
                      device: torch.device) -> dict[str, float]:
    network.eval()
    rows = []
    for episode in range(episodes):
        env = CoSREnv(num_stas_per_ap=10, fixed_topology=fixed_topology,
                      fixed_seed=scenario_seed if fixed_topology else None,
                      max_steps=warmup_steps + steps, arrival_rate_per_sta=arrival_rate, power_mode="pf_pair")
        state, info = env.reset(seed=seed + episode)
        for index in range(warmup_steps + steps):
            action = _greedy_action(network, state, np.asarray(info["action_mask"], dtype=bool), device)
            state, _, _, truncated, info = env.step(action)
            if index + 1 == warmup_steps:
                env.begin_measurement_window()
            if truncated:
                break
        rows.append(env.measurement_metrics())
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def train_cosr_dqn(*, episodes: int = 300, steps: int = 300, warmup_steps: int = 150,
                   replay_warmup: int = 2_000, batch_size: int = 128,
                   buffer_capacity: int = 50_000, learning_rate: float = 3e-4,
                   gamma: float = 0.99, target_update_interval: int = 250,
                   eval_interval: int = 25, eval_episodes: int = 3,
                   arrival_rate: float = 0.45, fixed_topology: bool = True,
                   expert_steps: int = 6_000, expert_pretrain_updates: int = 500,
                   imitation_weight: float = 0.30, seed: int = 0,
                   imitation_floor: float = 0.05,
                   lr_decay_after_episode: int | None = None,
                   lr_decay_gamma: float = 0.50,
                   scenario_seed: int = config.TRAIN_SCENARIO_SEED,
                   output_dir: str = "results_data/cosr_dqn") -> dict:
    """Train structured masked Double-DQN, warm-started by Group-Oldest + PF."""
    if min(episodes, steps, warmup_steps, batch_size, expert_steps) < 1:
        raise ValueError("episode, step, warmup, batch and expert counts must be positive")
    imitation_scale_for_episode(
        episode=0, episodes=episodes, imitation_weight=imitation_weight,
        imitation_floor=imitation_floor,
    )
    if lr_decay_after_episode is not None and lr_decay_after_episode < 0:
        raise ValueError("lr_decay_after_episode must be non-negative or None")
    if not 0.0 < lr_decay_gamma <= 1.0:
        raise ValueError("lr_decay_gamma must be in (0, 1]")
    set_seed(seed)
    rng = random.Random(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = CoSREnv(num_stas_per_ap=10, fixed_topology=fixed_topology,
                  fixed_seed=scenario_seed if fixed_topology else None,
                  max_steps=steps, arrival_rate_per_sta=arrival_rate, power_mode="pf_pair")
    state, _ = env.reset(seed=seed)
    online = GroupScoringNetwork(state.size, env.groups).to(device)
    target = GroupScoringNetwork(state.size, env.groups).to(device)
    optimizer = torch.optim.Adam(online.parameters(), lr=learning_rate)
    expert_env = CoSREnv(num_stas_per_ap=10, fixed_topology=fixed_topology,
                         fixed_seed=scenario_seed if fixed_topology else None,
                         max_steps=min(steps, 300), arrival_rate_per_sta=arrival_rate, power_mode="pf_pair")
    examples = collect_expert_examples(expert_env, steps=expert_steps, seed=50_000 + seed)
    for _ in range(expert_pretrain_updates):
        expert_update(online, optimizer, examples, batch_size, rng, device)
    target.load_state_dict(online.state_dict())
    replay = MaskedReplayBuffer(buffer_capacity)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    history, global_step, best_delay = [], 0, float("inf")

    for episode in range(episodes):
        # episode is zero-indexed, so 150 applies after the reported episode 150.
        if lr_decay_after_episode is not None and episode == lr_decay_after_episode:
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] *= lr_decay_gamma
        state, info = env.reset(seed=seed + episode)
        reward_sum = 0.0
        reward_throughput = 0.0
        reward_delay_cost = 0.0
        reward_queue_cost = 0.0
        invalid_actions = 0
        epsilon = max(0.02, 0.30 * (1.0 - episode / max(episodes * 0.60, 1)))
        imitation_scale = imitation_scale_for_episode(
            episode=episode, episodes=episodes, imitation_weight=imitation_weight,
            imitation_floor=imitation_floor,
        )
        online.train()
        for _ in range(steps):
            mask = np.asarray(info["action_mask"], dtype=bool)
            action = int(rng.choice(np.flatnonzero(mask))) if rng.random() < epsilon else _greedy_action(online, state, mask, device)
            next_state, reward, _, truncated, next_info = env.step(action)
            invalid_actions += int(next_info["invalid_action"])
            reward_terms = next_info["reward_terms"]
            reward_throughput += reward_terms["throughput"]
            reward_delay_cost += reward_terms["delay_cost"]
            reward_queue_cost += reward_terms["queue_cost"]
            replay.add(Transition(state.copy(), action, float(reward), next_state.copy(), bool(truncated),
                                  np.asarray(next_info["action_mask"], dtype=bool).copy()))
            state, info, reward_sum, global_step = next_state, next_info, reward_sum + reward, global_step + 1
            if len(replay) >= max(replay_warmup, batch_size):
                batch = replay.sample(batch_size, rng)
                states = torch.as_tensor(np.stack([x.state for x in batch]), dtype=torch.float32, device=device)
                actions = torch.as_tensor([x.action for x in batch], dtype=torch.long, device=device).unsqueeze(1)
                rewards = torch.as_tensor([x.reward for x in batch], dtype=torch.float32, device=device)
                next_states = torch.as_tensor(np.stack([x.next_state for x in batch]), dtype=torch.float32, device=device)
                dones = torch.as_tensor([x.done for x in batch], dtype=torch.float32, device=device)
                next_masks = torch.as_tensor(np.stack([x.next_mask for x in batch]), dtype=torch.bool, device=device)
                predicted = online(states).gather(1, actions).squeeze(1)
                with torch.no_grad():
                    next_actions = masked_argmax(online(next_states), next_masks)
                    next_values = target(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                    td_target = rewards + gamma * (1.0 - dones) * next_values
                loss = nn.functional.smooth_l1_loss(predicted, td_target)
                if imitation_scale:
                    expert_batch = rng.sample(examples, min(batch_size, len(examples)))
                    expert_states = torch.as_tensor(np.stack([x.state for x in expert_batch]), dtype=torch.float32, device=device)
                    expert_masks = torch.as_tensor(np.stack([x.mask for x in expert_batch]), dtype=torch.bool, device=device)
                    expert_actions = torch.as_tensor([x.action for x in expert_batch], dtype=torch.long, device=device)
                    loss = loss + imitation_scale * nn.functional.cross_entropy(
                        online(expert_states).masked_fill(~expert_masks, -torch.inf), expert_actions
                    )
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), 10.0)
                optimizer.step()
                if global_step % target_update_interval == 0:
                    target.load_state_dict(online.state_dict())
            if truncated:
                break
        row = {"episode": episode + 1, "reward": reward_sum, "epsilon": epsilon,
               "invalid_actions": invalid_actions, "reward_throughput": reward_throughput,
               "reward_delay_cost": reward_delay_cost, "reward_queue_cost": reward_queue_cost,
               "imitation_scale": imitation_scale,
               "learning_rate": optimizer.param_groups[0]["lr"]}
        if (episode + 1) % eval_interval == 0 or episode + 1 == episodes:
            evaluation = evaluate_cosr_dqn(online, episodes=eval_episodes, warmup_steps=warmup_steps,
                                            steps=steps, seed=10_000 + seed, arrival_rate=arrival_rate,
                                            fixed_topology=fixed_topology, scenario_seed=scenario_seed,
                                            device=device)
            row.update({f"eval_{key}": value for key, value in evaluation.items()})
            if evaluation["p95_delay_txops"] < best_delay:
                best_delay = evaluation["p95_delay_txops"]
                torch.save(online.state_dict(), output / "best_structured_dqn.pt")
                torch.save({
                    "episode": episode + 1,
                    "online_state_dict": online.state_dict(),
                    "target_state_dict": target.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "run_seed": seed,
                    "scenario_seed": scenario_seed,
                    "evaluation": evaluation,
                }, output / "best_structured_dqn_checkpoint.pt")
        history.append(row)
        if "eval_p95_delay_txops" in row:
            print(
                f"ep={episode + 1:04d} reward={reward_sum:.2f} eps={epsilon:.3f} "
                f"lr={row['learning_rate']:.1e} p95={row['eval_p95_delay_txops']:.2f} txop"
            )
    torch.save(online.state_dict(), output / "last_structured_dqn.pt")
    torch.save({
        "episode": episodes,
        "online_state_dict": online.state_dict(),
        "target_state_dict": target.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "run_seed": seed,
        "scenario_seed": scenario_seed,
    }, output / "last_structured_dqn_checkpoint.pt")
    return {"algorithm": "structured_central_masked_double_dqn", "seed": seed,
            "episodes": episodes, "steps": steps, "arrival_rate": arrival_rate,
            "fixed_topology": fixed_topology, "scenario_seed": scenario_seed,
            "expert": "group_oldest_pf", "imitation_weight": imitation_weight,
            "imitation_floor": imitation_floor,
            "lr_decay_after_episode": lr_decay_after_episode,
            "lr_decay_gamma": lr_decay_gamma, "history": history}
