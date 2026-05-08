import json
import math
import random

from util import FIELD_LENGTH_MM, FIELD_WIDTH_MM

DEFAULT_ACTIONS = [
    # ('stop', 0.0, 0.0, []),
    ('forward_slow', 0.5, 0.0, []),
    ('Forward_fast', 1, 0.0, []),
    ('Back_fast', -1, 0.0, []),
    ('Left_turn_fast', 0.0, -1, ['Right_turn_fast', ]),
    ('Right_turn_fast', 0.0, 1, ['Left_turn_fast', ]),
]


class QAgent:
    def __init__(self, team='red', actions=None, alpha=0.3, gamma=0.99,
                 epsilon=0.0, softmax=False):
        self.team = team
        self.actions = actions or DEFAULT_ACTIONS
        actIdx = {self.actions[i][0]: i for i in range(len(self.actions))}
        self.disallowed = [{actIdx.get(name, -1) for name in act[3]} for act in self.actions]
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.softmax = softmax
        self.q = {}
        self.counts = {}
        self.last_state = None
        self.last_action = None

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
            'axes': ['x', 'y', 'heading', 'dist', 'acc', 'prev', 'ballx', 'bally'],
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
        if action_index is None:
            return 0, 0
        return tuple(self.actions[action_index][1:3])

    def reset_episode(self):
        self.last_state = None
        self.last_action = None

    def remember(self, state, action):
        self.last_state = state
        self.last_action = action

    def learn_from_transition(
            self, next_state, reward, terminal=False, last_state=None,
            last_action=None, increment=None):
        increment = (0 if last_state is not None else 1) if increment is None else increment
        last_state = self.last_state if last_state is None else self.last_state
        last_action = self.last_action if last_action is None else self.last_action
        if last_state is None or last_action is None:
            return
        q, n = self.row(last_state)
        old_value = q[last_action]
        if terminal or not increment:
            target = reward
        else:
            target = reward + self.gamma * max(self.row(next_state)[0])
        alpha = max(self.alpha * 0.1, self.alpha / (1 + n[self.last_action] ** 0.5))
        q[last_action] = old_value + alpha * (target - old_value)
        n[last_action] += increment

    def weighted_choice(self, options, weights):
        r = random.random() * sum(weights)
        for opt, w in zip(options, weights):
            if r < w:
                return opt
            r -= w

    def choose_action(self, state, last_action):
        q, n = self.row(state)
        if last_action is None:
            act = list(range(len(self.actions)))
        else:
            act = [a for a in range(len(self.actions)) if a not in self.disallowed[last_action]]
        maxq = max([q[a] for a in act])
        if self.epsilon > 0:
            sumn = sum(n[a] for a in act)
            epsilon = max(self.epsilon * 0.2, self.epsilon / ((sumn + 1) ** 0.5))
            if random.random() < epsilon:
                w = [1 / (n[a] + 1) for a in act]
                return self.weighted_choice(act, w)
        if self.softmax:
            w = [math.exp(qv - maxq) for qv in q]
            return self.weighted_choice(range(len(self.actions)), w)
        best = [a for a in act if q[a] == maxq]
        return random.choice(best)

    def row(self, state):
        if state not in self.q:
            self.q[state] = [0] * len(self.actions)
            self.counts[state] = [0] * len(self.actions)
        return self.q[state], self.counts[state]

    def discretize(
            self, pose, distance_cm, reflectance_left, reflectance_right,
            previous_action, world=None):
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
        heading_bin = self.bin_value(((heading + 360 + 22.5) % 360.0), 0, 360, 16)
        distance_bin = self.distance_bin(distance_cm)
        confidence_bin = self.confidence_bin(std_x, std_y, std_heading)
        ball_bin = (0, 0)
        if world and 'ball' in world:
            ball_x_bin = self.bin_value(
                world['ball']['world_x_mm'], -FIELD_LENGTH_MM / 2, FIELD_LENGTH_MM / 2, 15)
            ball_y_bin = self.bin_value(
                world['ball']['world_y_mm'], -FIELD_WIDTH_MM / 2, FIELD_WIDTH_MM / 2, 7)
            ball_bin = (ball_x_bin, ball_y_bin)
        return '%d,%d,%d,%d,%d,%d,%d,%d' % (
            x_bin,
            y_bin,
            heading_bin,
            distance_bin,
            confidence_bin,
            previous_action or 0,
            ball_bin[0],
            ball_bin[1],
        )

    def bin_value(self, value, low, high, count):
        return min(max(0, int((value - low) * count / (high - low))), count - 1)

    def distance_bin(self, distance_cm):
        try:
            d = float(distance_cm)
        except Exception:
            d = 65535
        if d < 7.5:
            return 0
        if d < 11.6:
            return 1
        if d < 17.85:
            return 2
        if d < 27.5:
            return 3
        if d < 42.25:
            return 4
        if d < 65:
            return 5
        if d < 100:
            return 6
        return 7

    def confidence_bin(self, std_x, std_y, std_heading):
        position_std = max(std_x, std_y)
        if position_std < 100 and std_heading < 8:
            return 0
        if position_std < 200 and std_heading < 16:
            return 1
        if position_std < 400 and std_heading < 32:
            return 2
        return 3
