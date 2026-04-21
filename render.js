const CONFIG = CONFIG_JSON;

const canvas = document.getElementById('field');
const ctx = canvas.getContext('2d');

let scale = 1;
let gd = 0;
let robots = {};
let ball = { world_x_mm: 0, world_y_mm: 0, vel_x_mmps: 0, vel_y_mmps: 0 };
const joystickState = {};
const ws = new WebSocket(`ws://${location.host}/ws`);

function mmScale(mm) {
  return mm * scale;
}

function computeLayout() {
  scale = window.innerWidth / (CONFIG.field_length_mm + 2 * CONFIG.goal_depth_mm + 2);
  gd = Math.round(scale * CONFIG.goal_depth_mm);
  canvas.width = window.innerWidth;
  canvas.height = Math.round(scale * CONFIG.field_width_mm) + 2;
}

function fieldToCanvas(x, y) {
  return [(x + CONFIG.field_length_mm / 2) * scale + 1 + gd, (CONFIG.field_width_mm / 2 - y) * scale + 1];
}

function createElement(tag, className, textContent) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (textContent) el.textContent = textContent;
  return el;
}

function clampDeadzone(value, deadzone = 0.2) {
  if (Math.abs(value) < deadzone) return 0;
  if (Math.abs(value) > 1.0) return Math.sign(value);
  return value;
}

function drawCircle(context, x, y, radius, fillStyle, strokeStyle, lineWidth) {
  context.beginPath();
  context.arc(x, y, radius, 0, 2 * Math.PI);
  if (fillStyle) {
    context.fillStyle = fillStyle;
    context.fill();
  }
  if (strokeStyle) {
    context.strokeStyle = strokeStyle;
    context.lineWidth = lineWidth;
    context.stroke();
  }
}

function makeJoystick(robotId) {
  const size = 160;
  const knobR = 15;
  const maxR = size / 2 - knobR - 4;
  const cx = size / 2,
    cy = size / 2;

  const wrap = createElement('div', 'joystick-wrap');
  wrap.dataset.robotId = robotId;
  wrap.appendChild(createElement('div', 'joystick-label', robotId));

  const jc = createElement('canvas', 'joystick-canvas');
  jc.width = jc.height = size;
  wrap.appendChild(jc);

  const velLabel = createElement('div', 'joystick-label', 'L: 0  R: 0');
  wrap.appendChild(velLabel);

  const state = { dx: 0, dy: 0, active: false };
  joystickState[robotId] = { state, velLabel, jc, size, knobR };

  function drawJoystick() {
    const jctx = jc.getContext('2d');
    jctx.clearRect(0, 0, size, size);
    drawCircle(jctx, cx, cy, size / 2 - 2, '#2a2a2a', '#555', 1);
    jctx.beginPath();
    jctx.moveTo(cx, cy - maxR);
    jctx.lineTo(cx, cy + maxR);
    jctx.moveTo(cx - maxR, cy);
    jctx.lineTo(cx + maxR, cy);
    jctx.strokeStyle = '#444';
    jctx.lineWidth = 1;
    jctx.stroke();
    drawCircle(jctx, cx - state.dx * maxR, cy - state.dy * maxR, knobR, '#888', '#bbb', 1.5);
  }

  function getOffsetInCanvas(e) {
    const rect = jc.getBoundingClientRect();
    const clientX = e.clientX ?? e.touches[0].clientX;
    const clientY = e.clientY ?? e.touches[0].clientY;
    return [clientX - rect.left, clientY - rect.top];
  }

  function updateFromOffset(ox, oy) {
    state.dx = clampDeadzone(-(ox - cx) / maxR);
    state.dy = clampDeadzone(-(oy - cy) / maxR);
    sendVelocity(robotId, state.dx, state.dy, velLabel);
    drawJoystick();
  }

  function onStart(e) {
    if (e.type === 'touchstart') e.preventDefault();
    state.active = true;
    updateFromOffset(...getOffsetInCanvas(e));
  }

  function onMove(e) {
    if (!state.active) return;
    if (e.type === 'touchmove') e.preventDefault();
    updateFromOffset(...getOffsetInCanvas(e));
  }

  function release() {
    if (!state.active) return;
    state.active = false;
    state.dx = state.dy = 0;
    sendVelocity(robotId, 0, 0, velLabel);
    drawJoystick();
  }

  jc.addEventListener('mousedown', onStart);
  jc.addEventListener('touchstart', onStart, { passive: false });
  window.addEventListener('mousemove', onMove);
  window.addEventListener('touchmove', onMove, { passive: false });
  window.addEventListener('mouseup', release);
  window.addEventListener('touchend', release);

  drawJoystick();
  return wrap;
}

function sendVelocity(robotId, dx, dy, velLabel) {
  const maxV = CONFIG.max_wheel_speed_mmps;
  const vl = (dy - dx) * maxV;
  const vr = (dy + dx) * maxV;
  velLabel.textContent = `L: ${vl.toFixed(1)}  R: ${vr.toFixed(1)}`;
  if (ws.readyState === WebSocket.OPEN) {
    // ws.send(JSON.stringify({ type: 'cmd_vel', robot_id: robotId, cmd_vel_left: vl, cmd_vel_right: vr }));
    ws.send(JSON.stringify({ type: 'arcade', robot_id: robotId, straight: dy, turn: dx }));
  }
}

function rebuildJoysticks() {
  const container = document.getElementById('joysticks');
  const virtualIds = Object.keys(robots).filter((id) => robots[id].virtual);
  const existing = new Set([...container.querySelectorAll('.joystick-wrap')].map((el) => el.dataset.robotId));
  for (const id of virtualIds) {
    if (!existing.has(id)) container.appendChild(makeJoystick(id));
  }
  for (const el of container.querySelectorAll('.joystick-wrap')) {
    if (!virtualIds.includes(el.dataset.robotId)) {
      el.remove();
      delete joystickState[el.dataset.robotId];
    }
  }
}

function drawRoundedRect(context, x, y, halfW, halfH, r) {
  context.beginPath();
  context.moveTo(x - halfW + r, y - halfH);
  context.lineTo(x + halfW - r, y - halfH);
  context.arcTo(x + halfW, y - halfH, x + halfW, y - halfH + r, r);
  context.lineTo(x + halfW, y + halfH - r);
  context.arcTo(x + halfW, y + halfH, x + halfW - r, y + halfH, r);
  context.lineTo(x - halfW + r, y + halfH);
  context.arcTo(x - halfW, y + halfH, x - halfW, y + halfH - r, r);
  context.lineTo(x - halfW, y - halfH + r);
  context.arcTo(x - halfW, y - halfH, x - halfW + r, y - halfH, r);
  context.closePath();
}

function drawField() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.save();

  const fw = mmScale(CONFIG.field_length_mm);
  const fh = mmScale(CONFIG.field_width_mm);
  const r = mmScale(CONFIG.corner_radius_mm);
  const gw = mmScale(CONFIG.goal_width_mm);
  const ox = 1 + gd,
    oy = 1;
  const gTop = oy + (fh - gw) / 2;

  for (const line of CONFIG.tape_lines) {
    const [lx] = fieldToCanvas(line.x_mm, 0);
    ctx.strokeStyle = line.color;
    ctx.lineWidth = Math.max(1, Math.round(mmScale(CONFIG.tape_width_mm)));
    ctx.beginPath();
    ctx.moveTo(lx, oy);
    ctx.lineTo(lx, oy + fh);
    ctx.stroke();
  }

  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(ox + r, oy);
  ctx.lineTo(ox + fw - r, oy);
  ctx.arcTo(ox + fw, oy, ox + fw, oy + r, r);
  ctx.lineTo(ox + fw, gTop);
  ctx.moveTo(ox + fw, gTop + gw);
  ctx.lineTo(ox + fw, oy + fh - r);
  ctx.arcTo(ox + fw, oy + fh, ox + fw - r, oy + fh, r);
  ctx.lineTo(ox + r, oy + fh);
  ctx.arcTo(ox, oy + fh, ox, oy + fh - r, r);
  ctx.lineTo(ox, gTop + gw);
  ctx.moveTo(ox, gTop);
  ctx.lineTo(ox, oy + r);
  ctx.arcTo(ox, oy, ox + r, oy, r);
  ctx.stroke();

  for (const side of [-1, 1]) {
    const wallX = side === 1 ? ox + fw : ox;
    ctx.strokeStyle = '#aaaaaa';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(wallX, gTop);
    ctx.lineTo(wallX + side * gd, gTop);
    ctx.lineTo(wallX + side * gd, gTop + gw);
    ctx.lineTo(wallX, gTop + gw);
    ctx.stroke();
  }

  ctx.restore();
}

function drawBall() {
  const [cx, cy] = fieldToCanvas(ball.world_x_mm, ball.world_y_mm);
  ctx.save();
  drawCircle(ctx, cx, cy, mmScale(CONFIG.ball_diameter_mm / 2), '#f0f0f0', '#333333', 1);
  ctx.restore();
}

function drawRobots() {
  const len2 = mmScale(CONFIG.robot_length_mm / 2);
  const wid2 = mmScale(CONFIG.robot_width_mm / 2);
  const r = mmScale(CONFIG.robot_corner_radius_mm);
  for (const id in robots) {
    const rob = robots[id];
    if (rob.world_x_mm == null || rob.world_y_mm == null) continue;
    const [cx, cy] = fieldToCanvas(rob.world_x_mm, rob.world_y_mm);
    const rad = ((rob.world_heading_deg ?? 0) * Math.PI) / 180;

    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(-rad);
    drawRoundedRect(ctx, 0, 0, len2, wid2, r);
    ctx.fillStyle = id.toLowerCase().includes('red')
      ? '#cc3333'
      : id.toLowerCase().includes('blue')
        ? '#3366cc'
        : '#888888';
    ctx.fill();
    ctx.strokeStyle = rob.virtual ? '#ffff00' : '#ffffff';
    ctx.lineWidth = rob.virtual ? 2 : 1;
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(len2, 0);
    ctx.lineTo(len2 * 0.6, -wid2 * 0.5);
    ctx.lineTo(len2 * 0.6, wid2 * 0.5);
    ctx.closePath();
    ctx.fillStyle = '#ffffff';
    ctx.fill();
    ctx.restore();

    ctx.fillStyle = '#ffffff';
    ctx.font = `${Math.max(9, Math.round(mmScale(40)))}px monospace`;
    ctx.fillText(id, cx, cy);
  }
}

function render() {
  computeLayout();
  drawField();
  drawBall();
  drawRobots();
}

function updateBallInfo() {
  const fmt = (v) => (v ?? 0).toFixed(1);
  document.getElementById('ball-info').textContent =
    `x: ${fmt(ball.world_x_mm)} mm   y: ${fmt(ball.world_y_mm)} mm   vx: ${fmt(ball.vel_x_mmps)} mm/s   vy: ${fmt(ball.vel_y_mmps)} mm/s`;
}

function updateTable() {
  const tbody = document.getElementById('robot-tbody');
  tbody.innerHTML = '';
  for (const id in robots) {
    const rob = robots[id];
    const tr = document.createElement('tr');
    const fields = [
      id,
      rob.world_x_mm?.toFixed(0),
      rob.world_y_mm?.toFixed(0),
      rob.world_heading_deg?.toFixed(1),
      rob.distance_cm?.toFixed(1),
      rob.reflectance_left?.toFixed(3),
      rob.reflectance_right?.toFixed(3),
      rob.left_encoder?.toFixed(0),
      rob.right_encoder?.toFixed(0),
      rob.virtual ? 'yes' : 'no',
    ];
    for (const f of fields) {
      const td = document.createElement('td');
      td.textContent = f ?? '-';
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
}

function postJson(url, data) {
  fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) });
}

function getInputValue(id) {
  return parseFloat(document.getElementById(id).value);
}

function sendOverride() {
  postJson('/pose_override', {
    robot_id: document.getElementById('ov-id').value,
    world_x_mm: getInputValue('ov-x'),
    world_y_mm: getInputValue('ov-y'),
    world_heading_deg: getInputValue('ov-hdg'),
  });
}

function sendBallState() {
  postJson('/ball', {
    world_x_mm: getInputValue('ball-x'),
    world_y_mm: getInputValue('ball-y'),
    vel_x_mmps: getInputValue('ball-vx'),
    vel_y_mmps: getInputValue('ball-vy'),
  });
}

window.addEventListener('resize', render);

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  const handlers = {
    init: () => {
      robots = msg.robots;
      ball = msg.ball ?? ball;
      rebuildJoysticks();
    },
    telemetry: () => {
      robots[msg.data.robot_id] = { ...robots[msg.data.robot_id], ...msg.data };
    },
    ball: () => {
      ball = msg.data;
    },
    pose_override: () => {
      const d = msg.data;
      robots[d.robot_id] = {
        ...robots[d.robot_id],
        robot_id: d.robot_id,
        virtual: true,
        world_x_mm: d.world_x_mm,
        world_y_mm: d.world_y_mm,
        world_heading_deg: d.world_heading_deg,
      };
      rebuildJoysticks();
    },
    virtual_robots: () => {
      for (const [id, data] of Object.entries(msg.data)) robots[id] = { ...robots[id], ...data };
    },
  };
  handlers[msg.type]?.();
  render();
  updateBallInfo();
  updateTable();
};

render();
