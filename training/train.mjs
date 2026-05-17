#!/usr/bin/env node
// Hockey 1v1 - GPU Training Script (Node.js)
// Runs self-play REINFORCE training and saves weights to backend API.
// Usage: BACKEND_URL=... MODEL_NAME=... EPISODES=... node train.mjs

// ── Config from env ──
const BACKEND_URL = process.env.BACKEND_URL || 'https://hockey-api-zrey.onrender.com';
const MODEL_NAME = process.env.MODEL_NAME || 'gpu-trained';
const TOTAL_EPISODES = parseInt(process.env.EPISODES || '100000', 10);
const SAVE_INTERVAL = parseInt(process.env.SAVE_INTERVAL || '1000', 10);
const REPORT_INTERVAL = parseInt(process.env.REPORT_INTERVAL || '100', 10);
const MODEL_ID = process.env.MODEL_ID || ''; // if set, updates existing model

// ── Constants (same as frontend) ──
const RINK_WIDTH = 66;
const RINK_HEIGHT = 29;
const CORNER_RADIUS = 5;
const PLAYER_RADIUS = 2;
const MAX_SPEED = 14;
const MAX_FORCE = 28;
const DECEL_DISTANCE = 8;
const ARRIVAL_THRESHOLD = 0.5;
const MIN_SPEED = 0.3;
const WALL_RESTITUTION = 0.4;
const BACKCHECK_DEPTH = 16.5;
const GOAL_DEPTH = 5;
const GOAL_WIDTH = 10;
const STEAL_MAX_DIST = 8;
const STEAL_MIN_DIST = 4;
const STEAL_CHANCE_FAR = 0.2;
const STEAL_CHANCE_NEAR = 0.5;
const STEAL_LOCK_DURATION = 1.0;
const STEAL_DIR_BONUS = 0.25;
const STEAL_DIR_PENALTY = 0.15;
const RESTEAL_DELAY = 1.5;
const FIXED_DT = 1 / 60;
const GOAL_DISPLAY_TIME = 2.0;

// AI constants
const NUM_FEATURES = 13;
const NUM_ACTIONS = 10;
const NETWORK_SIZES = [NUM_FEATURES, 64, 32, NUM_ACTIONS];
const DECISION_BUDGET = 5;
const BUDGET_WINDOW = 600;
const BUDGET_REFILL_RATE = DECISION_BUDGET / BUDGET_WINDOW;
const MOVE_DIST = 12;
const MAX_STEPS = 900;
const GAMMA = 0.99;
const LR = 0.003;

// ── Vec2 utilities ──
function v2(x, y) { return { x, y }; }
function v2Add(a, b) { return { x: a.x + b.x, y: a.y + b.y }; }
function v2Sub(a, b) { return { x: a.x - b.x, y: a.y - b.y }; }
function v2Scale(v, s) { return { x: v.x * s, y: v.y * s }; }
function v2Length(v) { return Math.sqrt(v.x * v.x + v.y * v.y); }
function v2Normalize(v) {
  const len = v2Length(v);
  if (len < 1e-8) return { x: 0, y: 0 };
  return { x: v.x / len, y: v.y / len };
}
function v2Distance(a, b) { return v2Length(v2Sub(a, b)); }
function v2Dot(a, b) { return a.x * b.x + a.y * b.y; }

// ── Zone helpers ──
function getGoalZoneRect() {
  const goalY = (RINK_HEIGHT - GOAL_WIDTH) / 2;
  return { x: RINK_WIDTH - GOAL_DEPTH, y: goalY, w: GOAL_DEPTH, h: GOAL_WIDTH };
}
function getBackcheckZoneRect() {
  return { x: 0, y: 0, w: BACKCHECK_DEPTH, h: RINK_HEIGHT };
}
function pointInRect(p, r) {
  return p.x >= r.x && p.x <= r.x + r.w && p.y >= r.y && p.y <= r.y + r.h;
}

// ── State creation ──
function createInitialState(possession) {
  const carrierPos = v2(12, RINK_HEIGHT / 2);
  const defenderPos = v2(40, RINK_HEIGHT / 2);
  const bluePos = possession === 0 ? carrierPos : defenderPos;
  const redPos = possession === 1 ? carrierPos : defenderPos;
  return {
    players: [
      { pos: v2(bluePos.x, bluePos.y), vel: v2(0, 0), destination: null, team: 'blue', mustBackcheck: false, lockDirection: false, stealLockTimer: 0 },
      { pos: v2(redPos.x, redPos.y), vel: v2(0, 0), destination: null, team: 'red', mustBackcheck: false, lockDirection: false, stealLockTimer: 0 },
    ],
    possession,
    score: [0, 0],
    paused: false,
    goalMessage: null,
    goalTimer: 0,
    stealMessage: null,
    stealMessageTimer: 0,
  };
}

function resetAfterGoal(state, newPossession) {
  const carrierPos = v2(12, RINK_HEIGHT / 2);
  const defenderPos = v2(40, RINK_HEIGHT / 2);
  state.players[0].pos = newPossession === 0 ? v2(carrierPos.x, carrierPos.y) : v2(defenderPos.x, defenderPos.y);
  state.players[1].pos = newPossession === 1 ? v2(carrierPos.x, carrierPos.y) : v2(defenderPos.x, defenderPos.y);
  for (const p of state.players) {
    p.vel = v2(0, 0); p.destination = null; p.mustBackcheck = false; p.lockDirection = false; p.stealLockTimer = 0;
  }
  state.possession = newPossession;
  state.goalMessage = null;
  state.goalTimer = 0;
}

// ── Player physics ──
function stepPlayer(player, dt) {
  if (!player.destination) return;
  if (player.lockDirection) {
    player.pos = v2Add(player.pos, v2Scale(player.vel, dt));
    return;
  }
  const toTarget = v2Sub(player.destination, player.pos);
  const dist = v2Length(toTarget);
  if (dist < ARRIVAL_THRESHOLD) {
    const speed = v2Length(player.vel);
    if (speed < MIN_SPEED) { player.vel = v2(0, 0); player.destination = null; return; }
  }
  let desiredSpeed = MAX_SPEED;
  if (dist < DECEL_DISTANCE) desiredSpeed = MAX_SPEED * Math.max(dist / DECEL_DISTANCE, 0.05);
  const desiredDir = v2Normalize(toTarget);
  const desiredVel = v2Scale(desiredDir, desiredSpeed);
  let steering = v2Sub(desiredVel, player.vel);
  const steeringMag = v2Length(steering);
  if (steeringMag > MAX_FORCE) steering = v2Scale(v2Normalize(steering), MAX_FORCE);
  player.vel = v2Add(player.vel, v2Scale(steering, dt));
  const speed = v2Length(player.vel);
  if (speed > MAX_SPEED) player.vel = v2Scale(v2Normalize(player.vel), MAX_SPEED);
  player.pos = v2Add(player.pos, v2Scale(player.vel, dt));
}

// ── Wall collisions ──
function handleWallCollision(player) {
  const r = PLAYER_RADIUS;
  if (player.pos.x - r < 0) { player.pos.x = r; player.vel.x = Math.abs(player.vel.x) * WALL_RESTITUTION; }
  if (player.pos.x + r > RINK_WIDTH) { player.pos.x = RINK_WIDTH - r; player.vel.x = -Math.abs(player.vel.x) * WALL_RESTITUTION; }
  if (player.pos.y - r < 0) { player.pos.y = r; player.vel.y = Math.abs(player.vel.y) * WALL_RESTITUTION; }
  if (player.pos.y + r > RINK_HEIGHT) { player.pos.y = RINK_HEIGHT - r; player.vel.y = -Math.abs(player.vel.y) * WALL_RESTITUTION; }
  const corners = [
    v2(CORNER_RADIUS, CORNER_RADIUS),
    v2(RINK_WIDTH - CORNER_RADIUS, CORNER_RADIUS),
    v2(CORNER_RADIUS, RINK_HEIGHT - CORNER_RADIUS),
    v2(RINK_WIDTH - CORNER_RADIUS, RINK_HEIGHT - CORNER_RADIUS),
  ];
  const cornerChecks = [
    (p) => p.x < CORNER_RADIUS && p.y < CORNER_RADIUS,
    (p) => p.x > RINK_WIDTH - CORNER_RADIUS && p.y < CORNER_RADIUS,
    (p) => p.x < CORNER_RADIUS && p.y > RINK_HEIGHT - CORNER_RADIUS,
    (p) => p.x > RINK_WIDTH - CORNER_RADIUS && p.y > RINK_HEIGHT - CORNER_RADIUS,
  ];
  for (let i = 0; i < 4; i++) {
    if (cornerChecks[i](player.pos)) {
      const center = corners[i];
      const diff = v2Sub(player.pos, center);
      const dist = v2Length(diff);
      const maxDist = CORNER_RADIUS - r;
      if (dist > maxDist && dist > 0) {
        const dir = v2Normalize(diff);
        player.pos = v2Add(center, v2Scale(dir, maxDist));
        const velNormal = v2Dot(player.vel, dir);
        if (velNormal > 0) {
          const normalComp = v2Scale(dir, velNormal);
          player.vel = v2Sub(player.vel, v2Scale(normalComp, 1 + WALL_RESTITUTION));
        }
      }
    }
  }
}

// ── Player-player collision ──
function handlePlayerCollision(p0, p1) {
  const diff = v2Sub(p0.pos, p1.pos);
  const dist = v2Length(diff);
  const minDist = PLAYER_RADIUS * 2;
  if (dist < minDist && dist > 0.01) {
    const dir = v2Normalize(diff);
    const overlap = minDist - dist;
    p0.pos = v2Add(p0.pos, v2Scale(dir, overlap / 2));
    p1.pos = v2Sub(p1.pos, v2Scale(dir, overlap / 2));
    const relVel = v2Sub(p0.vel, p1.vel);
    const velAlongNormal = v2Dot(relVel, dir);
    if (velAlongNormal < 0) {
      const impulse = v2Scale(dir, velAlongNormal);
      p0.vel = v2Sub(p0.vel, impulse);
      p1.vel = v2Add(p1.vel, impulse);
    }
  }
}

// ── Scoring ──
function checkGoalScored(state) {
  const carrier = state.players[state.possession];
  if (carrier.mustBackcheck) return null;
  const zone = getGoalZoneRect();
  if (pointInRect(carrier.pos, zone)) return carrier.team;
  return null;
}

function updateBackcheck(player) {
  if (!player.mustBackcheck) return;
  const zone = getBackcheckZoneRect();
  if (pointInRect(player.pos, zone)) player.mustBackcheck = false;
}

// ── Steal ──
function attemptSteal(state, stealerIndex) {
  if (state.possession === stealerIndex) return { success: false };
  const stealer = state.players[stealerIndex];
  if (stealer.stealLockTimer > 0) return { success: false };
  const carrier = state.players[state.possession];
  const dist = v2Distance(stealer.pos, carrier.pos);
  stealer.lockDirection = true;
  stealer.stealLockTimer = STEAL_LOCK_DURATION;
  if (dist > STEAL_MAX_DIST) return { success: false };
  let chance;
  if (dist <= STEAL_MIN_DIST) { chance = STEAL_CHANCE_NEAR; }
  else { const t = (dist - STEAL_MIN_DIST) / (STEAL_MAX_DIST - STEAL_MIN_DIST); chance = STEAL_CHANCE_NEAR + t * (STEAL_CHANCE_FAR - STEAL_CHANCE_NEAR); }
  const carrierSpeed = v2Length(carrier.vel);
  if (carrierSpeed > MIN_SPEED) {
    const toStealer = v2Normalize(v2Sub(stealer.pos, carrier.pos));
    const carrierDir = v2Normalize(carrier.vel);
    const dot = v2Dot(carrierDir, toStealer);
    if (dot > 0) chance += dot * STEAL_DIR_BONUS;
    else chance += dot * STEAL_DIR_PENALTY;
    chance = Math.max(0.05, Math.min(0.85, chance));
  }
  if (Math.random() < chance) {
    const victimIdx = state.possession;
    state.possession = stealerIndex;
    stealer.mustBackcheck = true;
    state.players[victimIdx].stealLockTimer = RESTEAL_DELAY;
    return { success: true };
  }
  return { success: false };
}

// ── Main game step ──
function gameStep(state, dt) {
  if (state.goalMessage) {
    state.goalTimer -= dt;
    if (state.goalTimer <= 0) {
      const scoredByBlue = state.goalMessage.includes('BLUE');
      resetAfterGoal(state, scoredByBlue ? 1 : 0);
    }
    return;
  }
  if (state.stealMessage) {
    state.stealMessageTimer -= dt;
    if (state.stealMessageTimer <= 0) state.stealMessage = null;
  }
  for (const player of state.players) {
    if (player.stealLockTimer > 0) {
      player.stealLockTimer -= dt;
      if (player.stealLockTimer <= 0) { player.stealLockTimer = 0; player.lockDirection = false; }
    }
  }
  for (const player of state.players) stepPlayer(player, dt);
  for (const player of state.players) handleWallCollision(player);
  handlePlayerCollision(state.players[0], state.players[1]);
  const carrier = state.players[state.possession];
  updateBackcheck(carrier);
  const scored = checkGoalScored(state);
  if (scored) {
    const idx = scored === 'blue' ? 0 : 1;
    state.score[idx]++;
    state.goalMessage = `GOAL! ${scored.toUpperCase()} scores!`;
    state.goalTimer = GOAL_DISPLAY_TIME;
  }
}

// ── Neural Network ──
function randomWeight(fanIn) { return (Math.random() * 2 - 1) * Math.sqrt(2 / fanIn); }
function matVecMul(m, v) {
  const rows = m.length, cols = v.length, out = new Array(rows);
  for (let i = 0; i < rows; i++) { let s = 0; const row = m[i]; for (let j = 0; j < cols; j++) s += row[j] * v[j]; out[i] = s; }
  return out;
}
function vecAdd(a, b) { const out = new Array(a.length); for (let i = 0; i < a.length; i++) out[i] = a[i] + b[i]; return out; }
function relu(v) { const out = new Array(v.length); for (let i = 0; i < v.length; i++) out[i] = v[i] > 0 ? v[i] : 0; return out; }
function softmax(v) {
  const max = Math.max(...v);
  const exp = new Array(v.length);
  let sum = 0;
  for (let i = 0; i < v.length; i++) { exp[i] = Math.exp(v[i] - max); sum += exp[i]; }
  for (let i = 0; i < v.length; i++) exp[i] /= sum;
  return exp;
}

class PolicyNetwork {
  constructor(sizes) {
    this.layers = [];
    for (let i = 0; i < sizes.length - 1; i++) {
      const rows = sizes[i + 1], cols = sizes[i];
      const w = [];
      for (let r = 0; r < rows; r++) { w[r] = new Array(cols); for (let c = 0; c < cols; c++) w[r][c] = randomWeight(cols); }
      this.layers.push({ weights: w, biases: new Array(rows).fill(0) });
    }
  }
  forward(input) {
    const cache = { inputs: [input], preAct: [], postAct: [] };
    let cur = input;
    for (let l = 0; l < this.layers.length; l++) {
      const z = vecAdd(matVecMul(this.layers[l].weights, cur), this.layers[l].biases);
      cache.preAct.push(z);
      cur = l < this.layers.length - 1 ? relu(z) : softmax(z);
      cache.postAct.push(cur);
      if (l < this.layers.length - 1) cache.inputs.push(cur);
    }
    return { probs: cur, cache };
  }
  update(cache, action, advantage, lr) {
    const probs = cache.postAct[cache.postAct.length - 1];
    let delta = new Array(probs.length);
    for (let i = 0; i < probs.length; i++) delta[i] = (i === action ? 1 - probs[i] : -probs[i]) * advantage;
    for (let l = this.layers.length - 1; l >= 0; l--) {
      const inp = cache.inputs[l];
      const layer = this.layers[l];
      for (let i = 0; i < layer.weights.length; i++) {
        for (let j = 0; j < layer.weights[i].length; j++) layer.weights[i][j] += lr * delta[i] * inp[j];
        layer.biases[i] += lr * delta[i];
      }
      if (l > 0) {
        const prev = new Array(inp.length).fill(0);
        for (let j = 0; j < inp.length; j++) {
          for (let i = 0; i < delta.length; i++) prev[j] += layer.weights[i][j] * delta[i];
          if (cache.preAct[l - 1][j] <= 0) prev[j] = 0;
        }
        delta = prev;
      }
    }
  }
  serialize() { return JSON.stringify(this.layers); }
  static deserialize(json) {
    const net = new PolicyNetwork([1, 1]);
    net.layers = JSON.parse(json);
    return net;
  }
}

// ── State encoding ──
function encodeState(state, playerIdx) {
  const me = state.players[playerIdx];
  const opp = state.players[1 - playerIdx];
  const hasPuck = state.possession === playerIdx;
  const goalX = RINK_WIDTH - GOAL_DEPTH / 2;
  const goalY = RINK_HEIGHT / 2;
  return [
    me.pos.x / RINK_WIDTH, me.pos.y / RINK_HEIGHT,
    opp.pos.x / RINK_WIDTH, opp.pos.y / RINK_HEIGHT,
    me.vel.x / 14, me.vel.y / 14,
    opp.vel.x / 14, opp.vel.y / 14,
    hasPuck ? 1 : 0, me.mustBackcheck ? 1 : 0,
    (goalX - me.pos.x) / RINK_WIDTH, (goalY - me.pos.y) / RINK_HEIGHT,
    v2Distance(me.pos, opp.pos) / RINK_WIDTH,
  ];
}

// ── Action decoding ──
function decodeAction(action, state, playerIdx) {
  const me = state.players[playerIdx];
  const opp = state.players[1 - playerIdx];
  const r = PLAYER_RADIUS;
  const clamp = (p) => ({ x: Math.max(r, Math.min(RINK_WIDTH - r, p.x)), y: Math.max(r, Math.min(RINK_HEIGHT - r, p.y)) });
  switch (action) {
    case 0: return { destination: clamp(v2(RINK_WIDTH - GOAL_DEPTH / 2, RINK_HEIGHT / 2)), steal: false };
    case 1: return { destination: clamp(v2(opp.pos.x, opp.pos.y)), steal: false };
    case 2: return { destination: clamp(v2(BACKCHECK_DEPTH / 2, RINK_HEIGHT / 2)), steal: false };
    case 3: return { destination: clamp(v2(me.pos.x, me.pos.y - MOVE_DIST)), steal: false };
    case 4: return { destination: clamp(v2(me.pos.x, me.pos.y + MOVE_DIST)), steal: false };
    case 5: return { destination: clamp(v2(me.pos.x + MOVE_DIST, me.pos.y - MOVE_DIST * 0.7)), steal: false };
    case 6: return { destination: clamp(v2(me.pos.x + MOVE_DIST, me.pos.y + MOVE_DIST * 0.7)), steal: false };
    case 7: return { destination: clamp(v2(me.pos.x - MOVE_DIST, me.pos.y)), steal: false };
    case 8: return { destination: null, steal: true };
    case 9: default: return { destination: null, steal: false };
  }
}

// ── AI Agent ──
class AIAgent {
  constructor(network) {
    this.network = network;
    this.trajectory = [];
    this.lastAction = 9;
    this.explorationRate = 0.1;
    this.decisionTokens = DECISION_BUDGET;
    this.stepsSinceLastDecision = 0;
  }
  act(state, playerIdx) {
    this.decisionTokens = Math.min(DECISION_BUDGET, this.decisionTokens + BUDGET_REFILL_RATE);
    this.stepsSinceLastDecision++;
    if (this.decisionTokens < 1) return decodeAction(this.lastAction, state, playerIdx);
    if (this.stepsSinceLastDecision < 2) return decodeAction(this.lastAction, state, playerIdx);
    this.decisionTokens -= 1;
    this.stepsSinceLastDecision = 0;
    const features = encodeState(state, playerIdx);
    const { probs, cache } = this.network.forward(features);
    let action;
    if (Math.random() < this.explorationRate) {
      action = Math.floor(Math.random() * NUM_ACTIONS);
    } else {
      let r = Math.random();
      action = NUM_ACTIONS - 1;
      for (let i = 0; i < NUM_ACTIONS; i++) { r -= probs[i]; if (r <= 0) { action = i; break; } }
    }
    this.lastAction = action;
    this.trajectory.push({ features, action, cache, reward: 0 });
    return decodeAction(action, state, playerIdx);
  }
  reset() {
    this.trajectory = [];
    this.lastAction = 9;
    this.decisionTokens = DECISION_BUDGET;
    this.stepsSinceLastDecision = 0;
  }
}

// ── REINFORCE ──
function computeReturns(rewards, gamma) {
  const R = new Array(rewards.length);
  let G = 0;
  for (let i = rewards.length - 1; i >= 0; i--) { G = rewards[i] + gamma * G; R[i] = G; }
  return R;
}

function trainAgent(agent, gamma, lr) {
  const traj = agent.trajectory;
  if (traj.length === 0) return;
  const rewards = traj.map(s => s.reward);
  const returns = computeReturns(rewards, gamma);
  const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
  const variance = returns.reduce((a, b) => a + (b - mean) ** 2, 0) / returns.length;
  const std = Math.sqrt(variance) + 1e-8;
  for (let i = 0; i < traj.length; i++) {
    const advantage = (returns[i] - mean) / std;
    agent.network.update(traj[i].cache, traj[i].action, advantage, lr);
  }
}

// ── Episode runner ──
function runEpisode(network, agent0, agent1, episodeNum) {
  const possession = episodeNum % 2;
  const state = createInitialState(possession);
  agent0.reset();
  agent1.reset();
  let prevBlue = 0, prevRed = 0;
  for (let step = 0; step < MAX_STEPS; step++) {
    if (state.goalMessage) { gameStep(state, FIXED_DT); continue; }
    const a0 = agent0.act(state, 0);
    if (a0.destination) state.players[0].destination = a0.destination;
    if (a0.steal && state.possession !== 0) attemptSteal(state, 0);
    const a1 = agent1.act(state, 1);
    if (a1.destination) state.players[1].destination = a1.destination;
    if (a1.steal && state.possession !== 1) attemptSteal(state, 1);
    gameStep(state, FIXED_DT);
    if (state.score[0] > prevBlue) {
      addReward(agent0, 1); addReward(agent1, -1); prevBlue = state.score[0];
    }
    if (state.score[1] > prevRed) {
      addReward(agent1, 1); addReward(agent0, -1); prevRed = state.score[1];
    }
    if (!state.goalMessage) {
      const carrier = state.players[state.possession];
      const prog = 0.0005 * (carrier.pos.x / RINK_WIDTH);
      if (state.possession === 0 && agent0.trajectory.length > 0)
        agent0.trajectory[agent0.trajectory.length - 1].reward += prog;
      if (state.possession === 1 && agent1.trajectory.length > 0)
        agent1.trajectory[agent1.trajectory.length - 1].reward += prog;
    }
  }
  trainAgent(agent0, GAMMA, LR);
  trainAgent(agent1, GAMMA, LR);
  return { blue: state.score[0], red: state.score[1] };
}

function addReward(agent, r) {
  if (agent.trajectory.length > 0)
    agent.trajectory[agent.trajectory.length - 1].reward += r;
}

// ── HTTP helper ──
async function httpRequest(method, url, data) {
  const opts = {
    method,
    headers: { 'Content-Type': 'application/json' },
  };
  if (data) opts.body = JSON.stringify(data);
  const res = await fetch(url, opts);
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
  return res.json();
}

// ── Main training loop ──
async function main() {
  console.log(`Hockey 1v1 GPU Training`);
  console.log(`Backend: ${BACKEND_URL}`);
  console.log(`Model: ${MODEL_NAME}`);
  console.log(`Episodes: ${TOTAL_EPISODES}`);
  console.log(`Save every: ${SAVE_INTERVAL} episodes`);
  console.log('');

  const network = new PolicyNetwork(NETWORK_SIZES);
  const agent0 = new AIAgent(network);
  const agent1 = new AIAgent(network);

  let blueWins = 0, redWins = 0, draws = 0;
  let recentResults = [];
  let modelId = MODEL_ID ? parseInt(MODEL_ID, 10) : null;
  const startTime = Date.now();

  for (let ep = 1; ep <= TOTAL_EPISODES; ep++) {
    const result = runEpisode(network, agent0, agent1, ep);

    if (result.blue > result.red) { blueWins++; recentResults.push(1); }
    else if (result.red > result.blue) { redWins++; recentResults.push(-1); }
    else { draws++; recentResults.push(0); }
    if (recentResults.length > 100) recentResults.shift();

    // Decay exploration
    agent0.explorationRate = Math.max(0.02, 0.15 - ep * 0.0003);
    agent1.explorationRate = agent0.explorationRate;

    // Report progress
    if (ep % REPORT_INTERVAL === 0) {
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
      const epsPerSec = (ep / ((Date.now() - startTime) / 1000)).toFixed(1);
      const recentWins = recentResults.filter(r => r === 1).length;
      const winRate = (recentWins / recentResults.length * 100).toFixed(1);
      console.log(`[${elapsed}s] Episode ${ep}/${TOTAL_EPISODES} | B:${blueWins} R:${redWins} D:${draws} | WinRate:${winRate}% | ${epsPerSec} ep/s`);

      // Report stats to backend
      try {
        await httpRequest('POST', `${BACKEND_URL}/training/report`, {
          episode: ep,
          total_episodes: TOTAL_EPISODES,
          blue_wins: blueWins,
          red_wins: redWins,
          draws: draws,
          eps_per_sec: parseFloat(epsPerSec),
          model_name: MODEL_NAME,
        });
      } catch (e) {
        // Non-fatal - continue training even if reporting fails
      }
    }

    // Save weights periodically
    if (ep % SAVE_INTERVAL === 0) {
      const weights = network.serialize();
      try {
        if (modelId) {
          await httpRequest('PUT', `${BACKEND_URL}/models/${modelId}`, {
            weights, episodes: ep, blue_wins: blueWins, red_wins: redWins, draws,
          });
          console.log(`  -> Updated model #${modelId} at episode ${ep}`);
        } else {
          const saved = await httpRequest('POST', `${BACKEND_URL}/models`, {
            name: MODEL_NAME, weights, episodes: ep, blue_wins: blueWins, red_wins: redWins, draws,
          });
          modelId = saved.id;
          console.log(`  -> Created model #${modelId} "${MODEL_NAME}" at episode ${ep}`);
        }
      } catch (e) {
        console.error(`  -> Failed to save: ${e.message}`);
      }
    }
  }

  // Final save
  const weights = network.serialize();
  try {
    if (modelId) {
      await httpRequest('PUT', `${BACKEND_URL}/models/${modelId}`, {
        weights, episodes: TOTAL_EPISODES, blue_wins: blueWins, red_wins: redWins, draws,
      });
    } else {
      await httpRequest('POST', `${BACKEND_URL}/models`, {
        name: MODEL_NAME, weights, episodes: TOTAL_EPISODES, blue_wins: blueWins, red_wins: redWins, draws,
      });
    }
    console.log(`\nTraining complete! Final model saved.`);
  } catch (e) {
    console.error(`Failed final save: ${e.message}`);
  }

  // Signal completion
  try {
    await httpRequest('POST', `${BACKEND_URL}/training/report`, {
      episode: TOTAL_EPISODES,
      total_episodes: TOTAL_EPISODES,
      blue_wins: blueWins,
      red_wins: redWins,
      draws,
      eps_per_sec: 0,
      model_name: MODEL_NAME,
      status: 'completed',
    });
  } catch (e) { /* ignore */ }

  const totalTime = ((Date.now() - startTime) / 1000).toFixed(1);
  console.log(`Total time: ${totalTime}s | ${(TOTAL_EPISODES / ((Date.now() - startTime) / 1000)).toFixed(1)} ep/s`);
}

main().catch(e => { console.error(e); process.exit(1); });
