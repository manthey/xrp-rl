import math

GOAL_DEPTH_MM = 6 * 25.4
GOAL_WIDTH_MM = 20 * 25.4
FIELD_LENGTH_MM = 93 * 25.4 - GOAL_DEPTH_MM * 2
FIELD_WIDTH_MM = 45 * 25.4
CORNER_RADIUS_MM = 13.5 * 25.4
CORNER_BEVEL = 25.4
CORNER_MEET = (FIELD_LENGTH_MM / 2 - CORNER_RADIUS_MM + (CORNER_RADIUS_MM**2 - (
    FIELD_WIDTH_MM / 2 - CORNER_BEVEL - GOAL_WIDTH_MM / 2)**2)**0.5)
ROBOT_LENGTH_MM = 160
ROBOT_WIDTH_MM = 190
ROBOT_CORNER_RADIUS_MM = 20
ROBOT_DISTANCE_SENSOR_OFFSET = 70
ROBOT_DISTANCE_SENSOR_WIDTH_DEG = 15
ROBOT_REFLECTANCE_SENSOR_OFFSET = 60
ROBOT_REFLECTANCE_SENSOR_SIDE = 25 / 2
BALL_DIAMETER_MM = 2.25 * 25.4
BALL_RADIUS_MM = BALL_DIAMETER_MM / 2
TAPE_WIDTH_MM = 25

WHEEL_DIAMETER_MM = 60
WHEEL_BASE_MM = 155
TICKS_PER_REV = 585
MM_PER_TICK = math.pi * WHEEL_DIAMETER_MM / TICKS_PER_REV
MAX_WHEEL_SPEED_MMPS = 250.0
INERTIA_MMPS_PER_TICK = 50.0

TAPE_LINES = [
    {'x_mm': -FIELD_LENGTH_MM / 4, 'color': 'blue', 'rel': 0.23},
    {'x_mm': FIELD_LENGTH_MM / 4, 'color': 'red', 'rel': 0.24},
    {'x_mm': 0, 'color': 'white', 'rel': 0.20},
    {'x_mm': -FIELD_LENGTH_MM / 2 + 1, 'color': 'blue', 'rel': 0.23},
    {'x_mm': FIELD_LENGTH_MM / 2 - 1, 'color': 'red', 'rel': 0.24},
]
FIELD_CONFIG = {
    'field_length_mm': FIELD_LENGTH_MM,
    'field_width_mm': FIELD_WIDTH_MM,
    'goal_width_mm': GOAL_WIDTH_MM,
    'goal_depth_mm': GOAL_DEPTH_MM,
    'corner_radius_mm': CORNER_RADIUS_MM,
    'corner_bevel': CORNER_BEVEL,
    'corner_meet': CORNER_MEET,
    'robot_length_mm': ROBOT_LENGTH_MM,
    'robot_width_mm': ROBOT_WIDTH_MM,
    'robot_corner_radius_mm': ROBOT_CORNER_RADIUS_MM,
    'ball_diameter_mm': BALL_DIAMETER_MM,
    'tape_width_mm': TAPE_WIDTH_MM,
    'tape_lines': TAPE_LINES,
    'wheel_base_mm': WHEEL_BASE_MM,
    'max_wheel_speed_mmps': MAX_WHEEL_SPEED_MMPS,
}

HALF_LEN = FIELD_LENGTH_MM / 2
HALF_WID = FIELD_WIDTH_MM / 2
GOAL_HALF = GOAL_WIDTH_MM / 2
GOAL_BACK = HALF_LEN + GOAL_DEPTH_MM
INNER_X = HALF_LEN - CORNER_RADIUS_MM
INNER_Y = GOAL_HALF
CORNER_RADIUS_SQ = CORNER_RADIUS_MM * CORNER_RADIUS_MM

FIELD_BOUNDARY_SEGMENTS = []
for x1, y1, x2, y2 in [
    (-(CORNER_MEET - CORNER_BEVEL), HALF_WID, CORNER_MEET - CORNER_BEVEL, HALF_WID),
    (CORNER_MEET - CORNER_BEVEL, -HALF_WID, -(CORNER_MEET - CORNER_BEVEL), -HALF_WID),
    (-GOAL_BACK, -GOAL_HALF, -GOAL_BACK, GOAL_HALF),
    (-HALF_LEN, -GOAL_HALF, -GOAL_BACK, -GOAL_HALF),
    (-GOAL_BACK, GOAL_HALF, -HALF_LEN, GOAL_HALF),
    (GOAL_BACK, GOAL_HALF, GOAL_BACK, -GOAL_HALF),
    (GOAL_BACK, -GOAL_HALF, HALF_LEN, -GOAL_HALF),
    (HALF_LEN, GOAL_HALF, GOAL_BACK, GOAL_HALF),
    (CORNER_MEET - CORNER_BEVEL, HALF_WID, CORNER_MEET, HALF_WID - CORNER_BEVEL),
    (-CORNER_MEET, HALF_WID - CORNER_BEVEL, -CORNER_MEET + CORNER_BEVEL, HALF_WID),
    (CORNER_MEET, -HALF_WID + CORNER_BEVEL, CORNER_MEET - CORNER_BEVEL, -HALF_WID),
    (-CORNER_MEET + CORNER_BEVEL, -HALF_WID, -CORNER_MEET, -HALF_WID + CORNER_BEVEL),
]:
    FIELD_BOUNDARY_SEGMENTS.append((x1, y1, x2 - x1, y2 - y1))

FIELD_BOUNDARY_CORNERS = [
    (INNER_X, INNER_Y),
    (INNER_X, -INNER_Y),
    (-INNER_X, INNER_Y),
    (-INNER_X, -INNER_Y),
]


def ray_line_intersect(rx, ry, dx, dy, x1, y1, lx, ly):
    denom = dx * ly - dy * lx
    if abs(denom) < 1e-10:
        return None
    t = ((x1 - rx) * ly - (y1 - ry) * lx) / denom
    if t < 0:
        return None
    s = ((x1 - rx) * dy - (y1 - ry) * dx) / denom
    if 0 <= s <= 1:
        return t
    return None


def ray_circle_intersect(rx, ry, dx, dy, cx, cy, r):
    fx = rx - cx
    fy = ry - cy
    a = dx * dx + dy * dy
    b = 2 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - r * r
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    sqrt_disc = math.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2 * a)
    if t1 >= 0:
        return t1
    t2 = (-b + sqrt_disc) / (2 * a)
    if t2 >= 0:
        return t2
    return None


def ray_to_field_boundary(rx, ry, dx, dy):
    min_t = None
    for x1, y1, lx, ly in FIELD_BOUNDARY_SEGMENTS:
        t = ray_line_intersect(rx, ry, dx, dy, x1, y1, lx, ly)
        if t is not None:
            min_t = t if min_t is None else min(min_t, t)
    for cx, cy in FIELD_BOUNDARY_CORNERS:
        t = ray_circle_intersect(rx, ry, dx, dy, cx, cy, CORNER_RADIUS_MM)
        if t is not None:
            px = rx + dx * t
            py = ry + dy * t
            if abs(px) < CORNER_MEET:
                continue
            if (px - cx) * cx > 0 and (py - cy) * cy > 0:
                min_t = t if min_t is None else min(min_t, t)
    return min_t


def ray_to_ball(rx, ry, dx, dy, ball):
    bx = ball['world_x_mm']
    by = ball['world_y_mm']
    t = ray_circle_intersect(rx, ry, dx, dy, bx, by, BALL_RADIUS_MM)
    return t


def ray_to_robot(rx, ry, dx, dy, other_robot):
    ox = other_robot.get('world_x_mm')
    oy = other_robot.get('world_y_mm')
    if ox is None or oy is None:
        return None
    oh = math.radians(other_robot.get('world_heading_deg', 0.0))
    half_len = ROBOT_LENGTH_MM / 2
    half_wid = ROBOT_WIDTH_MM / 2
    points = robot_corners(ox, oy, oh, half_len, half_wid)
    min_t = None
    for px, py in points:
        vx = px - rx
        vy = py - ry
        proj = vx * dx + vy * dy
        if proj > 0:
            cross_dist = abs(vx * dy - vy * dx)
            if cross_dist < 50:
                min_t = proj if min_t is None else min(min_t, proj)
    robot_radius = math.sqrt(half_len**2 + half_wid**2)
    t = ray_circle_intersect(rx, ry, dx, dy, ox, oy, robot_radius)
    if t is not None:
        min_t = t if min_t is None else min(min_t, t)
    return min_t


def closest_point_on_rounded_rect(px, py, cx, cy, half_len, half_wid, corner_r, heading_rad):
    cos_h = math.cos(-heading_rad)
    sin_h = math.sin(-heading_rad)
    lx = (px - cx) * cos_h - (py - cy) * sin_h
    ly = (px - cx) * sin_h + (py - cy) * cos_h
    inner_half_len = half_len - corner_r
    inner_half_wid = half_wid - corner_r
    clamped_x = max(-inner_half_len, min(inner_half_len, lx))
    clamped_y = max(-inner_half_wid, min(inner_half_wid, ly))
    ox, oy = lx - clamped_x, ly - clamped_y
    olen = math.sqrt(ox * ox + oy * oy)
    if olen > 0:
        scale = corner_r / olen
        surface_lx = clamped_x + ox * scale
        surface_ly = clamped_y + oy * scale
        inside = olen < corner_r
    else:
        surface_lx = clamped_x + corner_r
        surface_ly = clamped_y
        inside = True
    cos_fwd = math.cos(heading_rad)
    sin_fwd = math.sin(heading_rad)
    wx = cx + surface_lx * cos_fwd - surface_ly * sin_fwd
    wy = cy + surface_lx * sin_fwd + surface_ly * cos_fwd
    return wx, wy, inside


def point_in_field(px, py):
    ax = abs(px)
    ay = abs(py)
    if ay > HALF_WID or ax > GOAL_BACK:
        return False
    if ax < CORNER_MEET - CORNER_BEVEL:
        return True
    if ay < INNER_Y and ax < HALF_LEN:
        return True
    if ax > HALF_LEN:
        return ay < GOAL_HALF
    if ax < CORNER_MEET:
        return HALF_WID - ay > ax - (CORNER_MEET - CORNER_BEVEL)
    sign_x = 1 if px >= 0 else -1
    sign_y = 1 if py >= 0 else -1
    corner_x = sign_x * INNER_X
    corner_y = sign_y * INNER_Y
    dx = px - corner_x
    dy = py - corner_y
    return dx * dx + dy * dy <= CORNER_RADIUS_SQ


def robot_corners(rx, ry, heading_rad, half_len, half_wid, corner_r=ROBOT_CORNER_RADIUS_MM):
    cos_h = math.cos(heading_rad)
    sin_h = math.sin(heading_rad)
    inner_half_len = half_len - corner_r
    inner_half_wid = half_wid - corner_r
    points = []
    straight_segments = [
        (-inner_half_len, -half_wid, inner_half_len, -half_wid),
        (half_len, -inner_half_wid, half_len, inner_half_wid),
        (inner_half_len, half_wid, -inner_half_len, half_wid),
        (-half_len, inner_half_wid, -half_len, -inner_half_wid),
    ]
    for x1, y1, x2, y2 in straight_segments:
        for t in (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0):
            lx = x1 + t * (x2 - x1)
            ly = y1 + t * (y2 - y1)
            wx = rx + lx * cos_h - ly * sin_h
            wy = ry + lx * sin_h + ly * cos_h
            points.append((wx, wy))
    corners = [
        (inner_half_len, inner_half_wid, 0, math.pi / 2),
        (inner_half_len, -inner_half_wid, -math.pi / 2, 0),
        (-inner_half_len, -inner_half_wid, math.pi, 3 * math.pi / 2),
        (-inner_half_len, inner_half_wid, math.pi / 2, math.pi),
    ]
    for cx, cy, start_ang, end_ang in corners:
        for t in (0.5,):
            ang = start_ang + t * (end_ang - start_ang)
            lx = cx + corner_r * math.cos(ang)
            ly = cy + corner_r * math.sin(ang)
            wx = rx + lx * cos_h - ly * sin_h
            wy = ry + lx * sin_h + ly * cos_h
            points.append((wx, wy))
    return points


def clamp_speed(v: float) -> float:
    return max(-MAX_WHEEL_SPEED_MMPS, min(MAX_WHEEL_SPEED_MMPS, v))
