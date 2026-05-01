import json
import math
import random

from util import FIELD_LENGTH_MM, FIELD_WIDTH_MM

DEFAULT_ACTIONS = [
    ('stop', 0.0, 0.0),  # must be 0-th action
    ('forward_slow', 0.5, 0.0),
    ('Forward_fast', 1, 0.0),
    ('Reverse_fast', -1, 0.0),
    ('left_turn', 0.0, -1),
    ('right_turn', 0.0, 1),
]


class QAgent:
    def __init__(self, team='red', actions=None, alpha=0.1, gamma=0.96,
                 epsilon=0.0, softmax=False):
        self.team = team
        self.actions = actions or DEFAULT_ACTIONS
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.softmax = softmax
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
        q, n = self.row(self.last_state)
        old_value = q[self.last_action]
        if terminal:
            target = reward
        else:
            target = reward + self.gamma * max(self.row(next_state)[0])
        alpha = max(self.alpha * 0.02, self.alpha / (1 + n[self.last_action]))
        q[self.last_action] = old_value + alpha * (target - old_value)
        n[self.last_action] += 1

    def weighted_choice(self, options, weights):
        r = random.random() * sum(weights)
        for opt, w in zip(options, weights):
            if r < w:
                return opt
            r -= w

    def choose_action(self, state):
        q, n = self.row(state)
        maxq = max(q)
        if self.softmax:
            w = [math.exp(qv - maxq) for qv in q]
            return self.weighted_choice(range(len(self.actions)), w)
        if self.epsilon > 0:
            adjn = [(1 - math.pow(0.995, nv)) / (1 - 0.995) for nv in n]
            sumn = sum(adjn)
            epsilon = max(self.epsilon * 0.1, self.epsilon / ((sumn + 1) ** 0.5))
            if random.random() < epsilon:
                w = [1 / (nv + 0.01) for nv in n]
                return self.weighted_choice(range(len(self.actions)), w)
        best = [i for i, value in enumerate(q) if value == maxq]
        return random.choice(best)

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
