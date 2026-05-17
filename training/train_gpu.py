#!/usr/bin/env python3
"""
Hockey 1v1 - Vectorized GPU Training with PyTorch
Runs N parallel environments on GPU using batched tensor operations.
All physics, neural network, and training run on GPU simultaneously.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import json
import os
import time
import urllib.request

# ── Config ──
BACKEND_URL = os.environ.get('BACKEND_URL', 'https://hockey-api-zrey.onrender.com')
MODEL_NAME = os.environ.get('MODEL_NAME', 'gpu-trained')
TOTAL_EPISODES = int(os.environ.get('EPISODES', '100000'))
SAVE_INTERVAL = int(os.environ.get('SAVE_INTERVAL', '1000'))
REPORT_INTERVAL = int(os.environ.get('REPORT_INTERVAL', '500'))
MODEL_ID = os.environ.get('MODEL_ID', '')
NUM_ENVS = int(os.environ.get('NUM_ENVS', '16384'))

# ── Physics Constants (matching frontend exactly) ──
RINK_W = 66.0
RINK_H = 29.0
CORNER_R = 5.0
PLAYER_R = 2.0
MAX_SPEED = 14.0
MAX_FORCE = 28.0
DECEL_DIST = 8.0
ARRIVAL_TH = 0.5
MIN_SPEED = 0.3
WALL_REST = 0.4
BACKCHECK_D = 16.5
GOAL_D = 5.0
GOAL_W = 10.0
STEAL_MAX_D = 8.0
STEAL_MIN_D = 4.0
STEAL_CHANCE_FAR = 0.2
STEAL_CHANCE_NEAR = 0.5
STEAL_LOCK_DUR = 1.0
STEAL_DIR_BONUS = 0.25
STEAL_DIR_PENALTY = 0.15
RESTEAL_DELAY = 1.5
DT = 1.0 / 60.0
GOAL_DISPLAY = 0.05  # minimal for training (instant reset)

# ── AI Constants ──
NUM_FEATURES = 13
NUM_ACTIONS = 10
DECISION_BUDGET = 5.0
BUDGET_WINDOW = 600.0
BUDGET_REFILL = DECISION_BUDGET / BUDGET_WINDOW
MOVE_DIST = 12.0
MAX_STEPS = 900
GAMMA = 0.99
LR = 0.003
INACTIVITY_TH = 300  # 5 seconds at 60fps
INACTIVITY_PEN = -0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ══════════════════════════════════════════════════════════════
# Vectorized Hockey Environment
# ══════════════════════════════════════════════════════════════

class VecHockeyEnv:
    """N parallel hockey games running on GPU as batched tensors."""

    def __init__(self, N):
        self.N = N
        # Player state: [N, 2(players), 2(xy)]
        self.pos = torch.zeros(N, 2, 2, device=device)
        self.vel = torch.zeros(N, 2, 2, device=device)
        self.dest = torch.zeros(N, 2, 2, device=device)
        self.has_dest = torch.zeros(N, 2, dtype=torch.bool, device=device)
        # Game state
        self.poss = torch.zeros(N, dtype=torch.long, device=device)
        self.must_bc = torch.zeros(N, 2, dtype=torch.bool, device=device)
        self.lock_dir = torch.zeros(N, 2, dtype=torch.bool, device=device)
        self.steal_tmr = torch.zeros(N, 2, device=device)
        self.score = torch.zeros(N, 2, dtype=torch.long, device=device)
        self.goal_tmr = torch.zeros(N, device=device)
        self.has_goal = torch.zeros(N, dtype=torch.bool, device=device)
        self.last_scorer = torch.zeros(N, dtype=torch.long, device=device)
        # Decision budget
        self.tokens = torch.full((N, 2), DECISION_BUDGET, device=device)
        self.steps_since = torch.zeros(N, 2, dtype=torch.long, device=device)
        self.last_act = torch.full((N, 2), 9, dtype=torch.long, device=device)
        # Episode tracking
        self.ep_num = torch.zeros(N, dtype=torch.long, device=device)
        # Pre-compute corner data
        self.corners = torch.tensor([
            [CORNER_R, CORNER_R],
            [RINK_W - CORNER_R, CORNER_R],
            [CORNER_R, RINK_H - CORNER_R],
            [RINK_W - CORNER_R, RINK_H - CORNER_R],
        ], device=device)
        self.carrier_pos_t = torch.tensor([12.0, RINK_H / 2], device=device)
        self.defender_pos_t = torch.tensor([40.0, RINK_H / 2], device=device)

    def reset(self, mask=None):
        if mask is None:
            mask = torch.ones(self.N, dtype=torch.bool, device=device)
        n = mask.sum().item()
        if n == 0:
            return
        poss = self.ep_num[mask] % 2
        cp = self.carrier_pos_t.unsqueeze(0).expand(n, -1)
        dp = self.defender_pos_t.unsqueeze(0).expand(n, -1)
        self.pos[mask, 0] = torch.where(poss.unsqueeze(-1) == 0, cp, dp)
        self.pos[mask, 1] = torch.where(poss.unsqueeze(-1) == 1, cp, dp)
        self.vel[mask] = 0
        self.has_dest[mask] = False
        self.poss[mask] = poss
        self.must_bc[mask] = False
        self.lock_dir[mask] = False
        self.steal_tmr[mask] = 0
        self.score[mask] = 0
        self.goal_tmr[mask] = 0
        self.has_goal[mask] = False
        self.tokens[mask] = DECISION_BUDGET
        self.steps_since[mask] = 0
        self.last_act[mask] = 9

    def get_obs(self):
        """Return observations [N, 2, 13] for both players."""
        obs = torch.zeros(self.N, 2, NUM_FEATURES, device=device)
        goal_x = RINK_W - GOAL_D / 2
        goal_y = RINK_H / 2
        for pi in range(2):
            oi = 1 - pi
            me_p = self.pos[:, pi]
            op_p = self.pos[:, oi]
            me_v = self.vel[:, pi]
            op_v = self.vel[:, oi]
            obs[:, pi, 0] = me_p[:, 0] / RINK_W
            obs[:, pi, 1] = me_p[:, 1] / RINK_H
            obs[:, pi, 2] = op_p[:, 0] / RINK_W
            obs[:, pi, 3] = op_p[:, 1] / RINK_H
            obs[:, pi, 4] = me_v[:, 0] / 14.0
            obs[:, pi, 5] = me_v[:, 1] / 14.0
            obs[:, pi, 6] = op_v[:, 0] / 14.0
            obs[:, pi, 7] = op_v[:, 1] / 14.0
            obs[:, pi, 8] = (self.poss == pi).float()
            obs[:, pi, 9] = self.must_bc[:, pi].float()
            obs[:, pi, 10] = (goal_x - me_p[:, 0]) / RINK_W
            obs[:, pi, 11] = (goal_y - me_p[:, 1]) / RINK_H
            obs[:, pi, 12] = torch.norm(me_p - op_p, dim=-1) / RINK_W
        return obs

    def apply_actions(self, actions, can_decide):
        """Decode actions [N,2] and update destinations/steals."""
        for pi in range(2):
            decided = can_decide[:, pi]
            act = actions[:, pi]
            me = self.pos[:, pi]
            opp = self.pos[:, 1 - pi]

            # Build all 8 movement destinations [N, 8, 2]
            d0 = torch.stack([torch.full((self.N,), RINK_W - GOAL_D / 2, device=device),
                              torch.full((self.N,), RINK_H / 2, device=device)], dim=-1)
            d1 = opp.clone()
            d2 = torch.stack([torch.full((self.N,), BACKCHECK_D / 2, device=device),
                              torch.full((self.N,), RINK_H / 2, device=device)], dim=-1)
            d3 = me + torch.tensor([0, -MOVE_DIST], device=device)
            d4 = me + torch.tensor([0, MOVE_DIST], device=device)
            d5 = me + torch.tensor([MOVE_DIST, -MOVE_DIST * 0.7], device=device)
            d6 = me + torch.tensor([MOVE_DIST, MOVE_DIST * 0.7], device=device)
            d7 = me + torch.tensor([-MOVE_DIST, 0], device=device)
            all_d = torch.stack([d0, d1, d2, d3, d4, d5, d6, d7], dim=1)  # [N,8,2]
            all_d[:, :, 0].clamp_(PLAYER_R, RINK_W - PLAYER_R)
            all_d[:, :, 1].clamp_(PLAYER_R, RINK_H - PLAYER_R)

            # Movement actions (0-7)
            is_move = decided & (act < 8)
            if is_move.any():
                idx = act.clamp(0, 7).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2)
                chosen = all_d.gather(1, idx).squeeze(1)
                self.dest[:, pi] = torch.where(is_move.unsqueeze(-1), chosen, self.dest[:, pi])
                self.has_dest[:, pi] = self.has_dest[:, pi] | is_move

            # Steal (action 8)
            is_steal = decided & (act == 8)
            if is_steal.any():
                self._steal_batch(pi, is_steal)

    def _steal_batch(self, si, mask):
        """Batch steal attempts for stealer index si."""
        not_carrier = self.poss != si
        no_timer = self.steal_tmr[:, si] <= 0
        can = mask & not_carrier & no_timer
        if not can.any():
            # Still set lock for attempts even if out of range
            self.lock_dir[:, si] = self.lock_dir[:, si] | (mask & not_carrier)
            self.steal_tmr[:, si] = torch.where(
                mask & not_carrier & no_timer,
                torch.full_like(self.steal_tmr[:, si], STEAL_LOCK_DUR),
                self.steal_tmr[:, si])
            return

        self.lock_dir[:, si] = self.lock_dir[:, si] | can
        self.steal_tmr[:, si] = torch.where(can,
            torch.full_like(self.steal_tmr[:, si], STEAL_LOCK_DUR),
            self.steal_tmr[:, si])

        sp = self.pos[:, si]
        # Carrier pos based on possession
        cp = torch.where(self.poss.unsqueeze(-1) == 0, self.pos[:, 0], self.pos[:, 1])
        dist = torch.norm(sp - cp, dim=-1)
        in_range = can & (dist <= STEAL_MAX_D)
        if not in_range.any():
            return

        t = ((dist - STEAL_MIN_D) / (STEAL_MAX_D - STEAL_MIN_D)).clamp(0, 1)
        chance = STEAL_CHANCE_NEAR + t * (STEAL_CHANCE_FAR - STEAL_CHANCE_NEAR)

        # Directional modifier
        cv = torch.where(self.poss.unsqueeze(-1) == 0, self.vel[:, 0], self.vel[:, 1])
        cspeed = torch.norm(cv, dim=-1)
        has_speed = cspeed > MIN_SPEED
        to_s = sp - cp
        to_s_n = to_s / (torch.norm(to_s, dim=-1, keepdim=True) + 1e-8)
        c_dir = cv / (cspeed.unsqueeze(-1) + 1e-8)
        dot = (c_dir * to_s_n).sum(dim=-1)
        d_mod = torch.where(dot > 0, dot * STEAL_DIR_BONUS, dot * STEAL_DIR_PENALTY)
        chance = torch.where(has_speed, (chance + d_mod).clamp(0.05, 0.85), chance)

        roll = torch.rand(self.N, device=device)
        success = in_range & (roll < chance)
        if not success.any():
            return

        victim = self.poss.clone()
        self.poss = torch.where(success, torch.full_like(self.poss, si), self.poss)
        self.must_bc[:, si] = self.must_bc[:, si] | success
        # Resteal delay on victim
        for vi in range(2):
            is_v = success & (victim == vi)
            self.steal_tmr[:, vi] = torch.where(is_v,
                torch.full_like(self.steal_tmr[:, vi], RESTEAL_DELAY),
                self.steal_tmr[:, vi])

    def step_physics(self):
        """Step all environments one frame."""
        # Goal timer countdown and reset
        self.goal_tmr = torch.where(self.has_goal, self.goal_tmr - DT, self.goal_tmr)
        reset_g = self.has_goal & (self.goal_tmr <= 0)
        if reset_g.any():
            self._reset_after_goal(reset_g)

        active = ~self.has_goal

        # Steal lock timers
        was_locked = self.steal_tmr > 0
        self.steal_tmr = (self.steal_tmr - DT).clamp(min=0)
        just_unlocked = was_locked & (self.steal_tmr <= 0)
        self.lock_dir = self.lock_dir & ~just_unlocked

        # Step players
        for pi in range(2):
            self._step_player(pi, active)

        # Wall collisions
        for pi in range(2):
            self._walls(pi, active)

        # Player-player collision
        self._player_collision(active)

        # Backcheck
        for pi in range(2):
            bc = self.must_bc[:, pi] & active
            in_zone = bc & (self.pos[:, pi, 0] <= BACKCHECK_D)
            self.must_bc[:, pi] = self.must_bc[:, pi] & ~in_zone

        # Goal check
        self._check_goals(active)

    def _step_player(self, pi, active):
        pos = self.pos[:, pi]
        vel = self.vel[:, pi]
        dest = self.dest[:, pi]
        has_d = self.has_dest[:, pi] & active
        locked = self.lock_dir[:, pi]

        # Locked: coast in current direction
        lm = has_d & locked
        pos_l = pos + vel * DT

        # Steering
        sm = has_d & ~locked
        to_t = dest - pos
        dist = torch.norm(to_t, dim=-1, keepdim=True).clamp(min=1e-8)

        # Arrival
        arrived = sm & (dist.squeeze(-1) < ARRIVAL_TH)
        spd = torch.norm(vel, dim=-1)
        stopped = arrived & (spd < MIN_SPEED)

        # Desired speed with decel
        ds = torch.where(dist.squeeze(-1) < DECEL_DIST,
                         MAX_SPEED * (dist.squeeze(-1) / DECEL_DIST).clamp(min=0.05),
                         torch.full((self.N,), MAX_SPEED, device=device))
        d_dir = to_t / dist
        d_vel = d_dir * ds.unsqueeze(-1)
        steer = d_vel - vel
        s_mag = torch.norm(steer, dim=-1, keepdim=True).clamp(min=1e-8)
        steer = torch.where(s_mag > MAX_FORCE, steer / s_mag * MAX_FORCE, steer)

        nv = vel + steer * DT
        ns = torch.norm(nv, dim=-1, keepdim=True).clamp(min=1e-8)
        nv = torch.where(ns > MAX_SPEED, nv / ns * MAX_SPEED, nv)
        np = pos + nv * DT

        # Combine
        f_pos = torch.where(lm.unsqueeze(-1), pos_l,
                torch.where(sm.unsqueeze(-1), np, pos))
        f_vel = torch.where(lm.unsqueeze(-1), vel,
                torch.where(sm.unsqueeze(-1), nv, vel))

        # Stopped
        f_vel = torch.where(stopped.unsqueeze(-1), torch.zeros_like(vel), f_vel)
        self.has_dest[:, pi] = self.has_dest[:, pi] & ~stopped

        self.pos[:, pi] = f_pos
        self.vel[:, pi] = f_vel

    def _walls(self, pi, active):
        p = self.pos[:, pi]
        v = self.vel[:, pi]
        R = PLAYER_R

        # Boundary clamp
        hit_l = active & (p[:, 0] < R)
        hit_r = active & (p[:, 0] > RINK_W - R)
        hit_t = active & (p[:, 1] < R)
        hit_b = active & (p[:, 1] > RINK_H - R)

        px = p[:, 0].clone()
        py = p[:, 1].clone()
        vx = v[:, 0].clone()
        vy = v[:, 1].clone()

        px = torch.where(hit_l, torch.full_like(px, R), px)
        vx = torch.where(hit_l, vx.abs() * WALL_REST, vx)
        px = torch.where(hit_r, torch.full_like(px, RINK_W - R), px)
        vx = torch.where(hit_r, -vx.abs() * WALL_REST, vx)
        py = torch.where(hit_t, torch.full_like(py, R), py)
        vy = torch.where(hit_t, vy.abs() * WALL_REST, vy)
        py = torch.where(hit_b, torch.full_like(py, RINK_H - R), py)
        vy = torch.where(hit_b, -vy.abs() * WALL_REST, vy)

        p_new = torch.stack([px, py], dim=-1)
        v_new = torch.stack([vx, vy], dim=-1)

        # Corner collisions
        corner_masks = [
            (p_new[:, 0] < CORNER_R) & (p_new[:, 1] < CORNER_R),
            (p_new[:, 0] > RINK_W - CORNER_R) & (p_new[:, 1] < CORNER_R),
            (p_new[:, 0] < CORNER_R) & (p_new[:, 1] > RINK_H - CORNER_R),
            (p_new[:, 0] > RINK_W - CORNER_R) & (p_new[:, 1] > RINK_H - CORNER_R),
        ]

        for ci in range(4):
            ic = active & corner_masks[ci]
            if not ic.any():
                continue
            center = self.corners[ci]
            diff = p_new - center
            dist = torch.norm(diff, dim=-1, keepdim=True).clamp(min=1e-8)
            max_d = CORNER_R - R
            too_far = ic & (dist.squeeze(-1) > max_d)
            if not too_far.any():
                continue
            d_n = diff / dist
            new_p = center + d_n * max_d
            v_norm = (v_new * d_n).sum(dim=-1, keepdim=True)
            reflects = too_far & (v_norm.squeeze(-1) > 0)
            n_comp = d_n * v_norm
            new_v = v_new - n_comp * (1 + WALL_REST)
            p_new = torch.where(too_far.unsqueeze(-1), new_p, p_new)
            v_new = torch.where(reflects.unsqueeze(-1), new_v, v_new)

        self.pos[:, pi] = p_new
        self.vel[:, pi] = v_new

    def _player_collision(self, active):
        diff = self.pos[:, 0] - self.pos[:, 1]
        dist = torch.norm(diff, dim=-1, keepdim=True).clamp(min=0.01)
        min_d = PLAYER_R * 2
        coll = active & (dist.squeeze(-1) < min_d)
        if not coll.any():
            return

        d_n = diff / dist
        overlap = min_d - dist
        sep = d_n * overlap / 2

        np0 = self.pos[:, 0] + sep
        np1 = self.pos[:, 1] - sep

        rv = self.vel[:, 0] - self.vel[:, 1]
        van = (rv * d_n).sum(dim=-1, keepdim=True)
        needs_imp = coll & (van.squeeze(-1) < 0)
        imp = d_n * van

        nv0 = torch.where(needs_imp.unsqueeze(-1), self.vel[:, 0] - imp, self.vel[:, 0])
        nv1 = torch.where(needs_imp.unsqueeze(-1), self.vel[:, 1] + imp, self.vel[:, 1])

        cu = coll.unsqueeze(-1)
        self.pos[:, 0] = torch.where(cu, np0, self.pos[:, 0])
        self.pos[:, 1] = torch.where(cu, np1, self.pos[:, 1])
        self.vel[:, 0] = torch.where(cu, nv0, self.vel[:, 0])
        self.vel[:, 1] = torch.where(cu, nv1, self.vel[:, 1])

    def _check_goals(self, active):
        cp = torch.where(self.poss.unsqueeze(-1) == 0, self.pos[:, 0], self.pos[:, 1])
        c_bc = torch.where(self.poss == 0, self.must_bc[:, 0], self.must_bc[:, 1])
        gy = (RINK_H - GOAL_W) / 2
        in_goal = (active & ~c_bc & ~self.has_goal &
                   (cp[:, 0] >= RINK_W - GOAL_D) &
                   (cp[:, 1] >= gy) & (cp[:, 1] <= gy + GOAL_W))
        if not in_goal.any():
            return
        self.score[:, 0] += (in_goal & (self.poss == 0)).long()
        self.score[:, 1] += (in_goal & (self.poss == 1)).long()
        self.has_goal = self.has_goal | in_goal
        self.goal_tmr = torch.where(in_goal,
            torch.full_like(self.goal_tmr, GOAL_DISPLAY), self.goal_tmr)
        self.last_scorer = torch.where(in_goal, self.poss, self.last_scorer)

    def _reset_after_goal(self, mask):
        new_p = torch.where(self.last_scorer[mask] == 0,
                            torch.ones(mask.sum(), dtype=torch.long, device=device),
                            torch.zeros(mask.sum(), dtype=torch.long, device=device))
        n = mask.sum().item()
        cp = self.carrier_pos_t.unsqueeze(0).expand(n, -1)
        dp = self.defender_pos_t.unsqueeze(0).expand(n, -1)
        self.pos[mask, 0] = torch.where(new_p.unsqueeze(-1) == 0, cp, dp)
        self.pos[mask, 1] = torch.where(new_p.unsqueeze(-1) == 1, cp, dp)
        self.vel[mask] = 0
        self.has_dest[mask] = False
        self.poss[mask] = new_p
        self.must_bc[mask] = False
        self.lock_dir[mask] = False
        self.steal_tmr[mask] = 0
        self.has_goal[mask] = False
        self.goal_tmr[mask] = 0


# ══════════════════════════════════════════════════════════════
# Policy Network (compatible with frontend weight format)
# ══════════════════════════════════════════════════════════════

class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(NUM_FEATURES, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, NUM_ACTIONS)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return F.softmax(self.fc5(x), dim=-1)

    def serialize_for_frontend(self):
        """Serialize weights in the same JSON format the frontend expects."""
        layers = []
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]:
            layers.append({
                "weights": layer.weight.detach().cpu().tolist(),
                "biases": layer.bias.detach().cpu().tolist(),
            })
        return json.dumps(layers)


# ══════════════════════════════════════════════════════════════
# HTTP helpers
# ══════════════════════════════════════════════════════════════

def http_post(url, data):
    body = json.dumps(data).encode()
    req = urllib.request.Request(url, data=body,
                                headers={'Content-Type': 'application/json'},
                                method='POST')
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def http_put(url, data):
    body = json.dumps(data).encode()
    req = urllib.request.Request(url, data=body,
                                headers={'Content-Type': 'application/json'},
                                method='PUT')
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


# ══════════════════════════════════════════════════════════════
# Training Loop
# ══════════════════════════════════════════════════════════════

def main():
    N = NUM_ENVS
    print(f"Hockey 1v1 - Vectorized GPU Training")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Parallel envs: {N}")
    print(f"Backend: {BACKEND_URL}")
    print(f"Model: {MODEL_NAME}")
    print(f"Total episodes: {TOTAL_EPISODES}")
    print(f"Save every: {SAVE_INTERVAL}")
    print()

    env = VecHockeyEnv(N)
    net = PolicyNet().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)

    total_ep = 0
    blue_wins = 0
    red_wins = 0
    draws = 0
    model_id = int(MODEL_ID) if MODEL_ID else None
    start_time = time.time()
    last_report = 0
    last_save = 0

    while total_ep < TOTAL_EPISODES:
        # ── New batch of N episodes ──
        env.ep_num = torch.arange(total_ep, total_ep + N, device=device, dtype=torch.long)
        env.reset()

        exploration_rate = max(0.02, 0.15 - total_ep * 0.0003)

        # Storage for REINFORCE (detached - no graph during rollout)
        stored_obs = []
        stored_actions = []
        stored_rewards = []
        stored_masks = []

        # ── Rollout: 900 steps with no_grad (physics + action selection) ──
        with torch.no_grad():
            for step in range(MAX_STEPS):
                # Decision budget
                env.tokens = (env.tokens + BUDGET_REFILL).clamp(max=DECISION_BUDGET)
                env.steps_since += 1
                can_decide = (env.tokens >= 1.0) & (env.steps_since >= 2)

                # Observations
                obs = env.get_obs()  # [N, 2, 13]
                obs_flat = obs.reshape(-1, NUM_FEATURES)  # [N*2, 13]

                # Forward pass
                probs = net(obs_flat)  # [N*2, 10]
                dist = Categorical(probs)
                new_acts = dist.sample()

                # Exploration
                explore = torch.rand(N * 2, device=device) < exploration_rate
                rand_acts = torch.randint(0, NUM_ACTIONS, (N * 2,), device=device)
                new_acts = torch.where(explore, rand_acts, new_acts)

                # Apply budget
                can_flat = can_decide.reshape(-1)
                last_flat = env.last_act.reshape(-1)
                eff_acts = torch.where(can_flat, new_acts, last_flat)

                # Spend tokens
                env.tokens = torch.where(can_decide, env.tokens - 1.0, env.tokens)
                env.steps_since = torch.where(can_decide,
                    torch.zeros_like(env.steps_since), env.steps_since)
                env.last_act = eff_acts.reshape(N, 2)

                # Store (detached)
                stored_obs.append(obs_flat.clone())
                stored_actions.append(eff_acts.clone())
                stored_masks.append(can_flat.float().clone())

                # Apply actions to environment
                env.apply_actions(eff_acts.reshape(N, 2), can_decide)

                # Step physics
                prev_score = env.score.clone()
                env.step_physics()

                # Compute rewards
                rewards = torch.zeros(N, 2, device=device)

                # Goal rewards
                sd = env.score - prev_score
                rewards[:, 0] += sd[:, 0].float() - sd[:, 1].float()
                rewards[:, 1] += sd[:, 1].float() - sd[:, 0].float()

                # Progress shaping
                carrier_x = torch.where(env.poss == 0,
                    env.pos[:, 0, 0], env.pos[:, 1, 0])
                prog = 0.0005 * carrier_x / RINK_W
                no_goal = (~env.has_goal).float()
                for pi in range(2):
                    rewards[:, pi] += prog * (env.poss == pi).float() * no_goal

                # Inactivity penalty
                for pi in range(2):
                    inactive = (env.steps_since[:, pi] > INACTIVITY_TH).float()
                    rewards[:, pi] += inactive * INACTIVITY_PEN

                stored_rewards.append(rewards.reshape(-1).clone())

        # ── Compute discounted returns ──
        rewards_t = torch.stack(stored_rewards)  # [900, N*2]
        returns = torch.zeros_like(rewards_t)
        G = torch.zeros(N * 2, device=device)
        for t in range(MAX_STEPS - 1, -1, -1):
            G = rewards_t[t] + GAMMA * G
            returns[t] = G

        # Normalize advantages
        adv = (returns - returns.mean()) / (returns.std() + 1e-8)

        # ── Policy update: re-compute log_probs with gradients ──
        all_obs = torch.stack(stored_obs)      # [900, N*2, 13]
        all_acts = torch.stack(stored_actions)  # [900, N*2]
        all_masks = torch.stack(stored_masks)   # [900, N*2]

        # Process in mini-batches of steps to control memory
        CHUNK = 100
        optimizer.zero_grad()
        total_decision_count = all_masks.sum()

        for s in range(0, MAX_STEPS, CHUNK):
            e = min(s + CHUNK, MAX_STEPS)
            chunk_obs = all_obs[s:e].reshape(-1, NUM_FEATURES)
            chunk_acts = all_acts[s:e].reshape(-1)
            chunk_adv = adv[s:e].reshape(-1)
            chunk_mask = all_masks[s:e].reshape(-1)

            probs = net(chunk_obs)
            dist = Categorical(probs)
            log_probs = dist.log_prob(chunk_acts)

            loss = -(log_probs * chunk_mask * chunk_adv.detach()).sum() / (total_decision_count + 1e-8)
            loss.backward()

        optimizer.step()

        # ── Count results ──
        fs = env.score
        bw = (fs[:, 0] > fs[:, 1]).sum().item()
        rw = (fs[:, 1] > fs[:, 0]).sum().item()
        d = N - bw - rw
        total_ep += N
        blue_wins += bw
        red_wins += rw
        draws += d

        # ── Report ──
        if total_ep - last_report >= max(REPORT_INTERVAL, N):
            last_report = total_ep
            elapsed = time.time() - start_time
            eps = total_ep / elapsed
            wr = blue_wins / max(1, blue_wins + red_wins + draws) * 100
            print(f"[{elapsed:.1f}s] Ep {total_ep}/{TOTAL_EPISODES} | "
                  f"B:{blue_wins} R:{red_wins} D:{draws} | "
                  f"WR:{wr:.1f}% | {eps:.0f} ep/s")
            try:
                http_post(f"{BACKEND_URL}/training/report", {
                    "episode": total_ep, "total_episodes": TOTAL_EPISODES,
                    "blue_wins": blue_wins, "red_wins": red_wins, "draws": draws,
                    "eps_per_sec": round(eps, 1), "model_name": MODEL_NAME,
                })
            except Exception:
                pass

        # ── Save ──
        if total_ep - last_save >= SAVE_INTERVAL:
            last_save = total_ep
            weights = net.serialize_for_frontend()
            try:
                if model_id:
                    http_put(f"{BACKEND_URL}/models/{model_id}", {
                        "weights": weights, "episodes": total_ep,
                        "blue_wins": blue_wins, "red_wins": red_wins, "draws": draws,
                    })
                    print(f"  -> Updated model #{model_id}")
                else:
                    result = http_post(f"{BACKEND_URL}/models", {
                        "name": MODEL_NAME, "weights": weights, "episodes": total_ep,
                        "blue_wins": blue_wins, "red_wins": red_wins, "draws": draws,
                    })
                    model_id = result.get("id")
                    print(f"  -> Created model #{model_id}")
            except Exception as e:
                print(f"  -> Save failed: {e}")

    # ── Final save ──
    weights = net.serialize_for_frontend()
    try:
        if model_id:
            http_put(f"{BACKEND_URL}/models/{model_id}", {
                "weights": weights, "episodes": TOTAL_EPISODES,
                "blue_wins": blue_wins, "red_wins": red_wins, "draws": draws,
            })
        else:
            http_post(f"{BACKEND_URL}/models", {
                "name": MODEL_NAME, "weights": weights, "episodes": TOTAL_EPISODES,
                "blue_wins": blue_wins, "red_wins": red_wins, "draws": draws,
            })
        print(f"\nTraining complete! Final model saved.")
    except Exception as e:
        print(f"Final save failed: {e}")

    # Signal completion
    try:
        http_post(f"{BACKEND_URL}/training/report", {
            "episode": TOTAL_EPISODES, "total_episodes": TOTAL_EPISODES,
            "blue_wins": blue_wins, "red_wins": red_wins, "draws": draws,
            "eps_per_sec": 0, "model_name": MODEL_NAME, "status": "completed",
        })
    except Exception:
        pass

    elapsed = time.time() - start_time
    print(f"Total: {elapsed:.1f}s | {TOTAL_EPISODES / elapsed:.0f} ep/s avg")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        print(f"FATAL ERROR: {err_msg}")
        try:
            http_post(f"{BACKEND_URL}/training/report", {
                "episode": 0, "total_episodes": 0,
                "blue_wins": 0, "red_wins": 0, "draws": 0,
                "eps_per_sec": 0, "model_name": MODEL_NAME,
                "status": f"error: {str(e)[:500]}",
            })
        except Exception:
            pass
        raise
