import json
import random

from util import FIELD_LENGTH_MM, FIELD_WIDTH_MM

DEFAULT_ACTIONS = [
    ('stop', 0.0, 0.0),  # must be 0-th action
    ('forward_slow', 0.5, 0.0),
    ('forward_fast', 1, 0.0),
    ('reverse_fast', -1, 0.0),
    ('turn_left', 0.0, -1),
    ('turn_right', 0.0, 1),
]


class QAgent:
    def __init__(self, team='red', actions=None, alpha=0.18, gamma=0.96, epsilon=0.0):
        self.team = team
        self.actions = actions or DEFAULT_ACTIONS
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q = {}
        self.counts = {}
        self.last_state = None
        self.last_action = 0

    def load(self, path):
        try:
            with open(path) as f:
                data = json.load(f)
            self.q = data.get('q', {})
            self.counts = data.get('counts', {})
        except Exception:
            self.q = {}
            self.counts = {}

    def save(self, path):
        data = {
            'actions': [list(a) for a in self.actions],
            'q': self.q,
            'counts': self.counts,
        }
        try:
            with open(path, 'w') as f:
                json.dump(data, f)
            return True
        except Exception:
            return False

    def command(self, action_index):
        return tuple(self.actions[action_index][1:])

    def reset_episode(self):
        self.last_state = None
        self.last_action = 0

    def remember(self, state, action):
        self.last_state = state
        self.last_action = action

    def learn_from_transition(self, next_state, reward, terminal=False):
        if self.last_state is None:
            return
        row, counts = self.row(self.last_state)
        old_value = row[self.last_action]
        if terminal:
            target = reward
        else:
            target = reward + self.gamma * max(self.row(next_state)[0])
        row[self.last_action] = old_value + self.alpha * (target - old_value)
        counts[self.last_action] += 1

    def choose_action(self, state):
        row, _ = self.row(state)
        if self.epsilon > 0 and random.random() < self.epsilon:
            return int(random.random() * len(self.actions))
        best_value = max(row)
        best = [i for i, value in enumerate(row) if value == best_value]
        return best[int(random.random() * len(best))]

    def row(self, state):
        if state not in self.q:
            self.q[state] = [0] * len(self.actions)
            self.counts[state] = [0] * len(self.actions)
        return self.q[state], self.counts[state]

    def discretize(self, pose, distance_cm, reflectance_left, reflectance_right, previous_action):
        x = float(pose.get('x_mm', 0.0))
        y = float(pose.get('y_mm', 0.0))
        heading = float(pose.get('heading_deg', 0.0))
        std_x = float(pose.get('std_x_mm', 999.0))
        std_y = float(pose.get('std_y_mm', 999.0))
        std_heading = float(pose.get('std_heading_deg', 999.0))
        if self.team == 'blue':
            x = -x
            y = -y
            heading = (180.0 + heading) % 360.0
        x_bin = self.bin_value(x, -FIELD_LENGTH_MM / 2, FIELD_LENGTH_MM / 2, 15)
        y_bin = self.bin_value(y, -FIELD_WIDTH_MM / 2, FIELD_WIDTH_MM / 2, 7)
        heading_bin = self.bin_value(((heading + 22.5) % 360.0), 0, 360, 8)
        distance_bin = self.distance_bin(distance_cm)
        confidence_bin = self.confidence_bin(std_x, std_y, std_heading)
        return '%d,%d,%d,%d,%d,%d' % (
            x_bin,
            y_bin,
            heading_bin,
            distance_bin,
            confidence_bin,
            previous_action
        )

    def bin_value(self, value, low, high, count):
        return min(max(0, int((value - low) * count / (high - low))), count - 1)

    def distance_bin(self, distance_cm):
        try:
            d = float(distance_cm)
        except Exception:
            return 4
        if d < 12.5:
            return 0
        if d < 25:
            return 1
        if d < 50:
            return 2
        if d < 100:
            return 3
        return 4

    def confidence_bin(self, std_x, std_y, std_heading):
        position_std = max(std_x, std_y)
        if position_std < 100 and std_heading < 8:
            return 0
        if position_std < 200 and std_heading < 16:
            return 1
        return 2


"""
Here’s a concrete, drop-in replacement for your `choose_action` method that intelligently uses visit counts while supporting both training and competitive modes.

### Recommended Implementation

 ```python
def choose_action(self, state, competitive=False):
    q_row, n_row = self.row(state)

    # --- Competitive Mode: Softmax (avoids predictability) ---
    if competitive:
        tau = self.competition_temp  # e.g., 0.5 for solo, 1.0 for 2v2
        q_shifted = q_row - np.max(q_row)  # Numerical stability
        exp_q = np.exp(q_shifted / tau)
        probs = exp_q / np.sum(exp_q)
        return np.random.choice(len(self.actions), p=probs)

    # --- Training Mode: Count-Based Intelligent Exploration ---
    # Dynamic epsilon: decays with state visits but never hits zero
    state_visits = n_row.sum()
    epsilon = max(self.epsilon_min, self.epsilon_base / np.sqrt(state_visits + 1))

    if random.random() < epsilon:
        # Explore: Weight random choice by inverse visit count
        # Prefer actions we've tried less (more uncertainty)
        explore_weights = 1.0 / (n_row + 0.01)  # +0.01 avoids division by zero
        explore_probs = explore_weights / explore_weights.sum()
        return np.random.choice(len(self.actions), p=explore_probs)

    # Exploit: Choose best Q-value, break ties randomly
    best_value = max(q_row)
    best_actions = [i for i, v in enumerate(q_row) if v == best_value]
    return random.choice(best_actions)
 ```

### Parameter Settings

 ```python
# Solo robot scoring (less non-stationary)
self.epsilon_base = 0.3
self.epsilon_min = 0.02
self.competition_temp = 0.5

# 2v2 soccer (highly non-stationary)
self.epsilon_base = 0.5
self.epsilon_min = 0.10  # Never drop below 10% exploration!
self.competition_temp = 1.0
 ```

### How It Works

1. **State-dependent epsilon**: Instead of a global `self.epsilon`, it calculates `epsilon` per state based on total visits. First visit to a state → ε ≈ 0.5; after 10,000 visits → ε ≈ 0.005 (but clamped at `epsilon_min`).

2. **Weighted exploration**: When exploring, actions with low `N(s,a)` get higher probability. This is far smarter than uniform random.

3. **Competitive softmax**: In 2v2, this prevents opponents from predicting your exact move when Q-values are close.

### Critical Integration: Recency-Weighted Counts

The `n_row` values should already incorporate forgetting. Update them in your learning step (not in `choose_action`):

 ```python
# In your Q-update method:
FORGET_RATE = 0.995  # Forget old visits (adjust 0.99-0.999 based on non-stationarity)
self.N[s][a] = FORGET_RATE * self.N[s][a] + 1
 ```

This ensures `n_row` reflects *recent* experience, making the exploration weights robust to sensor noise and opponent strategy shifts.


adjust_n = [(1 - math.pow(0.995, n) / (1 - 0.995) for n in n_row]
"""
