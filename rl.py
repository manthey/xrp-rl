import json
import random

from util import FIELD_LENGTH_MM, FIELD_WIDTH_MM

DEFAULT_ACTIONS = [
    ('stop', 0.0, 0.0),
    ('forward_slow', 0.5, 0.0),
    ('forward_fast', 1, 0.0),
    ('reverse_fast', -1, 0.0),
    ('turn_left', 0.0, 1),
    ('turn_right', 0.0, -1),
]


class QAgent:
    def __init__(self, team='red', actions=None, alpha=0.18, gamma=0.96, epsilon=0.0):
        self.team = team
        self.actions = actions or DEFAULT_ACTIONS
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q = {}
        self.last_state = None
        self.last_action = None

    def load(self, path):
        try:
            with open(path) as f:
                data = json.load(f)
            names = data.get('actions')
            if names:
                loaded = []
                for name in names:
                    for action in self.actions:
                        if action[0] == name:
                            loaded.append(action)
                            break
                if len(loaded) == len(names):
                    self.actions = loaded
            table = data.get('q', {})
            self.q = {}
            for state, values in table.items():
                self.q[state] = [float(v) for v in values]
        except Exception:
            self.q = {}

    def save(self, path):
        data = {
            'actions': [a[0] for a in self.actions],
            'q': self.q,
        }
        try:
            with open(path, 'w') as f:
                json.dump(data, f)
            return True
        except Exception:
            return False

    def command(self, action_index):
        action = self.actions[action_index]
        return action[1], action[2]

    def reset_episode(self):
        self.last_state = None
        self.last_action = None

    def remember(self, state, action):
        self.last_state = state
        self.last_action = action

    def learn_from_transition(self, next_state, reward, terminal=False):
        if self.last_state is None or self.last_action is None:
            return
        row = self.row(self.last_state)
        old_value = row[self.last_action]
        if terminal:
            target = reward
        else:
            target = reward + self.gamma * max(self.row(next_state))
        row[self.last_action] = old_value + self.alpha * (target - old_value)

    def choose_action(self, state):
        row = self.row(state)
        if self.epsilon > 0 and random.random() < self.epsilon:
            return int(random.random() * len(self.actions))
        best_value = max(row)
        best = []
        for i, value in enumerate(row):
            if value == best_value:
                best.append(i)
        return best[int(random.random() * len(best))]

    def row(self, state):
        values = self.q.get(state)
        if values is None:
            values = [0.0] * len(self.actions)
            self.q[state] = values
        if len(values) < len(self.actions):
            values.extend([0.0] * (len(self.actions) - len(values)))
        elif len(values) > len(self.actions):
            values = values[:len(self.actions)]
            self.q[state] = values
        return values

    def discretize(self, pose, distance_cm, reflectance_left, reflectance_right):
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

        x_bin = self.bin_value(x, -FIELD_LENGTH_MM / 2, FIELD_LENGTH_MM / 2, 10)
        y_bin = self.bin_value(y, -FIELD_WIDTH_MM / 2, FIELD_WIDTH_MM / 2, 6)
        heading_bin = self.bin_value(((heading + 22.5) % 360.0), 0, 360, 8)

        distance_bin = self.distance_bin(distance_cm)
        tape_bin = self.tape_bin(reflectance_left, reflectance_right)
        confidence_bin = self.confidence_bin(std_x, std_y, std_heading)

        return '%d,%d,%d,%d,%d,%d' % (
            x_bin,
            y_bin,
            heading_bin,
            distance_bin,
            tape_bin,
            confidence_bin,
        )

    def bin_value(self, value, low, high, count):
        if value <= low:
            return 0
        if value >= high:
            return count - 1
        return int((value - low) * count / (high - low))

    def distance_bin(self, distance_cm):
        try:
            d = float(distance_cm)
        except Exception:
            return 5
        if d < 8:
            return 0
        if d < 15:
            return 1
        if d < 30:
            return 2
        if d < 60:
            return 3
        if d < 100:
            return 4
        return 5

    def tape_bin(self, left, right):
        left_on = float(left) < 0.45
        right_on = float(right) < 0.45
        if left_on and right_on:
            return 3
        if left_on:
            return 1
        if right_on:
            return 2
        return 0

    def confidence_bin(self, std_x, std_y, std_heading):
        position_std = max(std_x, std_y)
        if position_std < 100 and std_heading < 8:
            return 0
        if position_std < 220 and std_heading < 18:
            return 1
        return 2
