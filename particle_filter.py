import math
import random

from util import (FIELD_LENGTH_MM, FIELD_WIDTH_MM, MM_PER_TICK,
                  ROBOT_DISTANCE_SENSOR_OFFSET,
                  ROBOT_REFLECTANCE_SENSOR_OFFSET,
                  ROBOT_REFLECTANCE_SENSOR_SIDE, TAPE_LINES, TAPE_WIDTH_MM,
                  WHEEL_BASE_MM, point_in_field, ray_to_field_boundary)

NUM_PARTICLES = 100
TAPE_THRESHOLD = 0.4
DISTANCE_INVALID = 6553.0
DISTANCE_SHORT_OBSTACLE_LIKELIHOOD = 0.4
RESAMPLE_FRACTION = 0.5
START_HEADING_SIGMA_DEG = 10.0
IMU_RELATIVE_SIGMA_DEG = 10.0

if hasattr(random, 'gauss'):
    gauss = random.gauss
else:
    def gauss(mu, sigma):
        u1 = random.random()
        u2 = random.random()
        if u1 < 1e-10:
            u1 = 1e-10
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return mu + sigma * z


class Particle:
    __slots__ = ('x', 'y', 'heading', 'weight', 'imu_offset')

    def __init__(self, x, y, heading, imu_offset, weight=1.0):
        self.x = x
        self.y = y
        self.heading = heading
        self.imu_offset = imu_offset
        self.weight = weight


class ParticleFilter:
    def __init__(self, team='red', num_particles=NUM_PARTICLES):
        self.num_particles = num_particles
        self.team = team
        self.dist_noise_mult = 0.2
        self.head_noise_mult = 0.1
        self.imu_drift_sigma = 0.02
        self.dist_sigma_mult = 0.05
        self.imu_weight = 0.9
        self.reset()

    def reset(self):
        self.particles = []
        self.prev_left_ticks = None
        self.prev_right_ticks = None
        self.prev_imu_deg = None
        self.best_x = 0.0
        self.best_y = 0.0
        self.best_heading = 0.0
        self.std_x = 1000.0
        self.std_y = 1000.0
        self.std_heading = 180.0
        self.initialize_particles()

    def initialize_particles(self):
        self.particles = []
        if self.team == 'red':
            x_min = -FIELD_LENGTH_MM / 2 + 100
            x_max = -FIELD_LENGTH_MM / 4
            facing = 0.0
        else:
            x_min = FIELD_LENGTH_MM / 4
            x_max = FIELD_LENGTH_MM / 2 - 100
            facing = 180.0
        y_min = -FIELD_WIDTH_MM / 2 + 100
        y_max = FIELD_WIDTH_MM / 2 - 100
        for _ in range(self.num_particles):
            x = random.uniform(x_min, x_max)
            y = random.uniform(y_min, y_max)
            heading = (facing + gauss(0, START_HEADING_SIGMA_DEG)) % 360
            self.particles.append(Particle(x, y, heading, 1.0 / self.num_particles))

    def predict(self, left_ticks, right_ticks, imu_deg):
        if self.prev_left_ticks is None:
            self.prev_left_ticks = left_ticks
            self.prev_right_ticks = right_ticks
            self.prev_imu_deg = imu_deg
            for p in self.particles:
                p.imu_offset = ((((p.heading - imu_deg) % 360) + 540) % 360) - 180
            return
        dl = (left_ticks - self.prev_left_ticks) * MM_PER_TICK
        dr = (right_ticks - self.prev_right_ticks) * MM_PER_TICK
        self.prev_left_ticks = left_ticks
        self.prev_right_ticks = right_ticks
        dist = (dl + dr) / 2.0
        d_heading_enc = math.degrees((dr - dl) / WHEEL_BASE_MM)
        d_imu = ((((imu_deg - self.prev_imu_deg) % 360) + 540) % 360) - 180

        self.prev_imu_deg = imu_deg
        d_heading = self.imu_weight * d_imu + (1 - self.imu_weight) * d_heading_enc
        dist_noise = max(2.0, abs(dist) * self.dist_noise_mult)
        heading_noise = max(0.5, abs(d_heading) * self.head_noise_mult)
        for p in self.particles:
            d = dist + gauss(0, dist_noise)
            dh = d_heading + gauss(0, heading_noise)
            p.imu_offset += gauss(0, self.imu_drift_sigma)
            mid = math.radians(p.heading + dh / 2.0)
            p.x += d * math.cos(mid)
            p.y += d * math.sin(mid)
            p.heading = (p.heading + dh) % 360

    def expected_distance(self, p):
        hr = math.radians(p.heading)
        dx = math.cos(hr)
        dy = math.sin(hr)
        sx = p.x + dx * ROBOT_DISTANCE_SENSOR_OFFSET
        sy = p.y + dy * ROBOT_DISTANCE_SENSOR_OFFSET
        if not point_in_field(sx, sy):
            return 0
        return ray_to_field_boundary(sx, sy, dx, dy)

    def reflectance_sensor_positions(self, p):
        hr = math.radians(p.heading)
        dx = math.cos(hr)
        dy = math.sin(hr)
        cx = p.x + dx * ROBOT_REFLECTANCE_SENSOR_OFFSET
        cy = p.y + dy * ROBOT_REFLECTANCE_SENSOR_OFFSET
        lx = cx - dy * ROBOT_REFLECTANCE_SENSOR_SIDE
        ly = cy + dx * ROBOT_REFLECTANCE_SENSOR_SIDE
        rx = cx + dy * ROBOT_REFLECTANCE_SENSOR_SIDE
        ry = cy - dx * ROBOT_REFLECTANCE_SENSOR_SIDE
        return lx, ly, rx, ry

    def expected_on_tape(self, sx, sy):
        if not point_in_field(sx, sy):
            return False
        for t in TAPE_LINES:
            if abs(sx - t['x_mm']) < TAPE_WIDTH_MM / 2:
                return True
        return False

    def update_weights(self, distance_cm, reflectance_left, reflectance_right):
        distance_mm = distance_cm * 10.0 if distance_cm < 6500 else None
        distance_valid = distance_mm is not None and distance_mm < DISTANCE_INVALID
        observed_left_tape = reflectance_left < TAPE_THRESHOLD
        observed_right_tape = reflectance_right < TAPE_THRESHOLD
        total = 0.0
        for p in self.particles:
            if not point_in_field(p.x, p.y):
                p.weight = 1e-9
                total += p.weight
                continue
            w = 1.0
            exp_dist = self.expected_distance(p)
            if distance_valid:
                if exp_dist is None or exp_dist > DISTANCE_INVALID:
                    w *= 0.3
                else:
                    err = distance_mm - exp_dist
                    sigma = max(20.0, exp_dist * self.dist_sigma_mult)
                    wall_likelihood = math.exp(-0.5 * (err / sigma) ** 2)
                    if distance_mm < exp_dist - 2 * sigma:
                        w *= max(wall_likelihood, DISTANCE_SHORT_OBSTACLE_LIKELIHOOD)
                    else:
                        w *= max(wall_likelihood, 0.05)
            else:
                if exp_dist is not None and exp_dist < 800:
                    w *= 0.05
            lx, ly, rx, ry = self.reflectance_sensor_positions(p)
            for sx, sy, observed in ((lx, ly, observed_left_tape),
                                     (rx, ry, observed_right_tape)):
                exp_tape = self.expected_on_tape(sx, sy)
                if observed and exp_tape:
                    w *= 5.0
                elif observed and not exp_tape:
                    w *= 0.05
                elif not observed and exp_tape:
                    w *= 0.05  # 0.4
            imu_heading = (self.prev_imu_deg + p.imu_offset) % 360
            err = ((((p.heading - imu_heading) % 360) + 540) % 360) - 180
            w *= math.exp(-0.5 * (err / IMU_RELATIVE_SIGMA_DEG) ** 2)
            p.weight = max(1e-9, w)
            total += p.weight
        if total > 0:
            for p in self.particles:
                p.weight /= total

    def effective_sample_size(self):
        s = 0.0
        for p in self.particles:
            s += p.weight * p.weight
        if s == 0:
            return 0
        return 1.0 / s

    def resample(self):
        ess = self.effective_sample_size()
        if ess > self.num_particles * RESAMPLE_FRACTION:
            return
        new_particles = []
        n = self.num_particles
        keep_random = max(1, n // 20)
        n_resample = n - keep_random
        step = 1.0 / n_resample
        u = random.uniform(0, step)
        c = self.particles[0].weight
        i = 0
        for j in range(n_resample):
            uj = u + j * step
            while uj > c and i < n - 1:
                i += 1
                c += self.particles[i].weight
            src = self.particles[i]
            new_particles.append(Particle(
                src.x + gauss(0, 5.0),
                src.y + gauss(0, 5.0),
                (src.heading + gauss(0, 1.0)) % 360,
                src.imu_offset,
                1.0 / n))
        best = max(self.particles, key=lambda p: p.weight)
        for _ in range(keep_random):
            new_particles.append(Particle(
                best.x + gauss(0, 50),
                best.y + gauss(0, 50),
                (best.heading + gauss(0, 10)) % 360,
                best.imu_offset,
                1.0 / n))
        self.particles = new_particles

    def estimate(self):
        sx = 0.0
        sy = 0.0
        sin_h = 0.0
        cos_h = 0.0
        wsum = 0.0
        for p in self.particles:
            sx += p.x * p.weight
            sy += p.y * p.weight
            hr = math.radians(p.heading)
            sin_h += math.sin(hr) * p.weight
            cos_h += math.cos(hr) * p.weight
            wsum += p.weight
        if wsum == 0:
            wsum = 1.0
        self.best_x = sx / wsum
        self.best_y = sy / wsum
        self.best_heading = math.degrees(math.atan2(sin_h, cos_h)) % 360
        vx = 0.0
        vy = 0.0
        vh = 0.0
        for p in self.particles:
            vx += p.weight * (p.x - self.best_x) ** 2
            vy += p.weight * (p.y - self.best_y) ** 2
            dh = ((p.heading - self.best_heading + 540) % 360) - 180
            vh += p.weight * dh * dh
        self.std_x = math.sqrt(vx / wsum)
        self.std_y = math.sqrt(vy / wsum)
        self.std_heading = math.sqrt(vh / wsum)
        return self.best_x, self.best_y, self.best_heading

    def step(self, left_ticks, right_ticks, distance_cm,
             reflectance_left, reflectance_right, imu_deg):
        self.predict(left_ticks, right_ticks, imu_deg)
        self.update_weights(distance_cm, reflectance_left, reflectance_right)
        self.estimate()
        self.resample()
        return self.best_x, self.best_y, self.best_heading

    def get_pose_with_error(self):
        return {
            'x_mm': self.best_x,
            'y_mm': self.best_y,
            'heading_deg': self.best_heading,
            'std_x_mm': self.std_x,
            'std_y_mm': self.std_y,
            'std_heading_deg': self.std_heading,
        }
