const CONFIG = CONFIG_JSON;

const canvas = document.getElementById('field');
const ctx = canvas.getContext('2d');

let scale = 1;
let gd = 0;
let robots = {};
let estimatedPoses = {};
let ball = { world_x_mm: 0, world_y_mm: 0, vel_x_mmps: 0, vel_y_mmps: 0 };
const joystickState = {};
let ws = null;

let qFiles = [];
let qIndex = -1;
let qvIndex = 0;
let qData = null;
let qAverages = null;
let qGridInfo = null;
let qCanvas = null;

function connectWebSocket() {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onmessage = onWsMessage;
  ws.onclose = () => {};
}

function closeWebSocket() {
  if (ws && ws.readyState === WebSocket.OPEN) ws.close();
}

document.addEventListener('visibilitychange', () => {
  if (document.hidden) closeWebSocket();
  else connectWebSocket();
});

function mmScale(mm) {
  return mm * scale;
}

function computeLayout() {
  const panelHeight = document.getElementById('panel').offsetHeight;
  const availHeight = window.innerHeight - panelHeight;
  const totalW = CONFIG.field_length_mm + 2 * CONFIG.goal_depth_mm + 2;
  const totalH = CONFIG.field_width_mm + 2;
  const scaleW = window.innerWidth / totalW;
  const scaleH = availHeight / totalH;
  scale = Math.min(scaleW, scaleH);
  const minScale = window.innerWidth / (2 * totalW);
  scale = Math.max(scale, minScale);
  gd = Math.round(scale * CONFIG.goal_depth_mm);
  if (canvas.width !== Math.round(scale * totalW) || canvas.height !== Math.round(scale * totalH)) {
    canvas.width = Math.round(scale * totalW);
    canvas.height = Math.round(scale * totalH);
    canvas.style.width = canvas.width + 'px';
    canvas.style.height = canvas.height + 'px';
  }
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
    if (e.type === 'touchstart') {
      e.preventDefault();
    }
    state.active = true;
    updateFromOffset(...getOffsetInCanvas(e));
  }

  function onMove(e) {
    if (!state.active) {
      return;
    }
    if (e.type === 'touchmove') {
      e.preventDefault();
    }
    updateFromOffset(...getOffsetInCanvas(e));
  }

  function release() {
    if (!state.active) {
      return;
    }
    state.active = false;
    state.dx = state.dy = 0;
    sendVelocity(robotId, 0, 0, velLabel);
    drawJoystick();
  }

  jc.addEventListener('mousedown', onStart);
  jc.addEventListener('touchstart', onStart, { passive: false, capture: true });
  window.addEventListener('mousemove', onMove);
  window.addEventListener('touchmove', onMove, { passive: false, capture: true });
  window.addEventListener('mouseup', release);
  window.addEventListener('touchend', release, { passive: false });

  drawJoystick();
  return wrap;
}

function sendVelocity(robotId, dx, dy, velLabel) {
  const maxV = CONFIG.max_wheel_speed_mmps;
  const vl = (dy - dx) * maxV;
  const vr = (dy + dx) * maxV;
  velLabel.textContent = `L: ${vl.toFixed(1)}  R: ${vr.toFixed(1)}`;
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'arcade', robot_id: robotId, straight: dy, turn: dx }));
  }
}

function train(active) {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'train', active: !!active }));
  }
}

async function hideExtended() {
  const btn = document.getElementById('hide-ext');
  const div = document.getElementById('extended');
  div.classList.toggle('hidden');
  btn.textContent = div.classList.contains('hidden') ? 'Show' : 'Hide';
}

async function toggleQState() {
  const btn = document.getElementById('show-qstate');
  if (qFiles.length === 0) {
    try {
      const res = await fetch('/q_files/list');
      const data = await res.json();
      qFiles = data.files || [];
    } catch (e) {
      return;
    }
  }
  if (qFiles.length === 0) return;

  if (qIndex < 0) {
    qIndex = 0;
    qvIndex = 0;
  } else {
    qvIndex = (qvIndex + 1) % 4;
    if (!qvIndex) {
      qIndex = (qIndex + 1) % (qFiles.length + 1);
    }
    if (qIndex === qFiles.length) {
      qIndex = -1;
      qData = null;
      qAverages = null;
      qCanvas = null;
      btn.textContent = 'Visualize: Off';
      render();
      return;
    }
  }
  const desc = ['', ' (dist)', ' (acc)', ' (prev)'];
  btn.textContent = `Visualize: ${qFiles[qIndex]}${desc[qvIndex]}`;
  try {
    const res = await fetch(`/q_files/${qIndex}`);
    qData = await res.json();
    processQData(qData);
    qCanvas = null;
    render();
  } catch (e) {
    qData = null;
    qCanvas = null;
  }
}

function processQData(data) {
  qAverages = {};
  let maxX = 0,
    maxY = 0,
    maxH = 0,
    maxP1 = 0;
  const sums = {};
  for (const key in data.q) {
    const parts = key.split(',').map(Number);
    const x = parts[0];
    const y = parts[1];
    const h = parts[2];
    const p1 = qvIndex === 0 ? 0 : parts[2 + qvIndex];
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
    if (h > maxH) maxH = h;
    if (p1 > maxP1) maxP1 = p1;
    const subKey = `${x},${y},${h},${p1}`;
    if (!sums[subKey]) {
      sums[subKey] = { q: data.q[key].map(() => 0), c: 0 };
    }
    const qVals = data.q[key];
    for (let i = 0; i < qVals.length; i++) {
      sums[subKey].q[i] += qVals[i];
    }
    sums[subKey].c++;
  }
  let maxMaxVal;
  let minMaxVal;
  for (const key in sums) {
    const avg = sums[key].q.map((v) => v / sums[key].c);
    let maxIdx = 0;
    let maxVal = -Infinity;
    for (let i = 0; i < avg.length; i++) {
      if (avg[i] > maxVal) {
        maxVal = avg[i];
        maxIdx = i;
      }
    }
    if (maxMaxVal === undefined) {
      minMaxVal = maxMaxVal = maxVal;
    }
    if (maxVal > maxMaxVal) {
      maxMaxVal = maxVal;
    }
    if (maxVal < minMaxVal) {
      minMaxVal = maxVal;
    }
    qAverages[key] = [maxIdx, maxVal];
  }
  qGridInfo = { maxX, maxY, maxH, maxP1, minMaxVal, maxMaxVal };
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

  const fw2 = mmScale(CONFIG.field_length_mm / 2);
  const fh2 = mmScale(CONFIG.field_width_mm / 2);
  const r = mmScale(CONFIG.corner_radius_mm);
  const gw2 = mmScale(CONFIG.goal_width_mm / 2);
  const cm = mmScale(CONFIG.corner_meet);
  const cb = mmScale(CONFIG.corner_bevel);
  const ox = 1 + gd + fw2;
  const oy = 1 + fh2;

  for (const line of CONFIG.tape_lines) {
    const [lx] = fieldToCanvas(line.x_mm, 0);
    ctx.strokeStyle = line.color;
    ctx.lineWidth = Math.max(1, Math.round(mmScale(CONFIG.tape_width_mm)));
    ctx.beginPath();
    ctx.moveTo(lx, oy - fh2);
    ctx.lineTo(lx, oy + fh2);
    ctx.stroke();
  }

  for (const sx of [-1, 1]) {
    for (const sy of [-1, 1]) {
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(ox, oy + sy * fh2);
      ctx.lineTo(ox + sx * (cm - cb), oy + sy * fh2);
      ctx.lineTo(ox + sx * cm, oy + sy * (fh2 - cb));
      const ang = Math.atan2(sy * (fh2 - cb - gw2), -sx * (fw2 - r - cm));
      ctx.arc(ox + sx * (fw2 - r), oy + sy * gw2, r, ang, sx > 0 ? 0 : Math.PI, sx * sy > 0);

      ctx.lineTo(ox + sx * (fw2 + gd), oy + sy * gw2);
      ctx.lineTo(ox + sx * (fw2 + gd), oy);
      ctx.stroke();
    }
  }

  ctx.restore();
}

function renderQStateOffscreen() {
  if (!qData || !qAverages) {
    qCanvas = null;
    return;
  }
  if (!qCanvas) qCanvas = document.createElement('canvas');
  qCanvas.width = canvas.width;
  qCanvas.height = canvas.height;
  const qCtx = qCanvas.getContext('2d');
  qCtx.clearRect(0, 0, qCanvas.width, qCanvas.height);

  const { maxX, maxY, maxH, maxP1, minMaxVal, maxMaxVal } = qGridInfo;
  const maxValRange = maxMaxVal - minMaxVal || 1;
  const binW = CONFIG.field_length_mm / (maxX + 1);
  const binH = CONFIG.field_width_mm / (maxY + 1);
  const maxR = Math.min(binW, binH) * 0.48;
  const innerR = maxR * 0.2;
  const ringW = (maxR - innerR) / (maxP1 + 1);
  const dTheta = (2 * Math.PI) / (maxH + 1);

  qCtx.save();
  for (let x = 0; x <= maxX; x++) {
    for (let y = 0; y <= maxY; y++) {
      const wx = (x + 0.5) * binW - CONFIG.field_length_mm / 2;
      const wy = (y + 0.5) * binH - CONFIG.field_width_mm / 2;
      const [cx, cy] = fieldToCanvas(wx, wy);

      for (let h = 0; h <= maxH; h++) {
        const theta = h * dTheta;
        const ca = -theta;
        for (let p1 = 0; p1 <= maxP1; p1++) {
          const subkey = `${x},${y},${h},${p1}`;
          if (qAverages[subkey] === undefined) {
            continue;
          }
          const [actionIdx, maxVal] = qAverages[subkey];
          const action = qData.actions[actionIdx];
          const r1 = scale * (innerR + p1 * ringW);
          const r2 = scale * (innerR + (p1 + 1) * ringW);
          const rMid = ((2 / 3) * (r2 ** 3 - r1 ** 3)) / (r2 ** 2 - r1 ** 2);

          qCtx.beginPath();
          qCtx.arc(cx, cy, r2, ca - dTheta / 2, ca + dTheta / 2);
          qCtx.arc(cx, cy, r1, ca + dTheta / 2, ca - dTheta / 2, true);
          const relMaxVal = ((maxVal - minMaxVal) / maxValRange) * 0.6 + 0.2;
          qCtx.fillStyle = `rgba(50, 50, 50, ${relMaxVal})`;
          qCtx.fill();

          const vx = action[1];
          const vy = action[2];
          const mag = Math.sqrt(vx * vx + vy * vy);
          if (mag > 0) {
            const len = Math.max(2, Math.min(scale * ringW, rMid * dTheta) * 0.9) * mag;
            const ux = vx / mag;
            const uy = vy / mag;
            const dx = (ux * Math.cos(ca) - uy * Math.sin(ca)) * len;
            const dy = (ux * Math.sin(ca) + uy * Math.cos(ca)) * len;
            const midX = cx + Math.cos(ca) * rMid;
            const midY = cy + Math.sin(ca) * rMid;
            const tipX = midX + dx / 2;
            const tipY = midY + dy / 2;
            const startX = midX - dx / 2;
            const startY = midY - dy / 2;
            const backX = midX - dx / 6;
            const backY = midY - dy / 6;
            const nx = -dy / 3;
            const ny = dx / 3;
            qCtx.beginPath();
            qCtx.moveTo(startX, startY);
            qCtx.lineTo(midX, midY);
            qCtx.strokeStyle = '#fff';
            qCtx.lineWidth = 1.5;
            qCtx.stroke();
            qCtx.beginPath();
            qCtx.moveTo(tipX, tipY);
            qCtx.lineTo(backX + nx, backY + ny);
            qCtx.lineTo(backX - nx, backY - ny);
            qCtx.closePath();
            qCtx.fillStyle = '#fff';
            qCtx.fill();
          }
        }
      }
    }
  }
  qCtx.restore();
}

function drawBall() {
  const [cx, cy] = fieldToCanvas(ball.world_x_mm, ball.world_y_mm);
  ctx.save();
  drawCircle(ctx, cx, cy, mmScale(CONFIG.ball_diameter_mm / 2), '#f0f0f0', '#333333', 1);
  ctx.restore();
}

function drawEstimatedPose(ctx, est) {
  const k = 2.0;
  ctx.save();
  const [ex, ey] = fieldToCanvas(est.x_mm, est.y_mm);
  ctx.translate(ex, ey);
  ctx.strokeStyle = 'rgba(0, 200, 255, 0.8)';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.ellipse(0, 0, k * mmScale(est.std_x_mm), k * mmScale(est.std_y_mm), 0, 0, 2 * Math.PI);
  ctx.closePath();
  ctx.stroke();
  const h = (est.heading_deg * Math.PI) / 180;
  const dh = (k * (est.std_heading_deg * Math.PI)) / 180;
  ctx.fillStyle = 'rgba(0, 200, 255, 0.8)';
  ctx.beginPath();
  ctx.moveTo(0, 0);
  ctx.ellipse(0, 0, k * mmScale(est.std_x_mm), k * mmScale(est.std_y_mm), 0, -h - dh, -h + dh);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function drawRobots() {
  const len2 = mmScale(CONFIG.robot_length_mm / 2);
  const wid2 = mmScale(CONFIG.robot_width_mm / 2);
  const r = mmScale(CONFIG.robot_corner_radius_mm);
  for (const id in robots) {
    const rob = robots[id];
    if (rob.world_x_mm === null || rob.world_y_mm === null) continue;
    const [cx, cy] = fieldToCanvas(rob.world_x_mm, rob.world_y_mm);
    const rad = ((rob.world_heading_deg ?? 0) * Math.PI) / 180;

    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(-rad);
    drawRoundedRect(ctx, 0, 0, len2, wid2, r);
    ctx.fillStyle = rob.team === 'blue' ? '#3333cc' : '#cc3333';
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

    if (estimatedPoses[id]) {
      drawEstimatedPose(ctx, estimatedPoses[id]);
    }

    ctx.fillStyle = '#ffffff';
    ctx.font = `${Math.max(9, Math.round(mmScale(36)))}px monospace`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(id, cx, cy);
  }
}

function render() {
  computeLayout();
  if (qData && (!qCanvas || qCanvas.width !== canvas.width || qCanvas.height !== canvas.height)) {
    renderQStateOffscreen();
  }
  drawField();
  if (qCanvas) {
    ctx.drawImage(qCanvas, 0, 0);
  }
  drawBall();
  drawRobots();
}

function updateBallInfo() {
  const fmt = (v) => (v ?? 0).toFixed(1);
  document.getElementById('ball-info').textContent = `x: ${fmt(ball.world_x_mm)} mm   y: ${fmt(
    ball.world_y_mm,
  )} mm   vx: ${fmt(ball.vel_x_mmps)} mm/s   vy: ${fmt(ball.vel_y_mmps)} mm/s`;
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
      rob.imu_heading_deg?.toFixed(1),
      rob.distance_cm?.toFixed(1),
      rob.reflectance_left?.toFixed(3),
      rob.reflectance_right?.toFixed(3),
      rob.left_encoder?.toFixed(0),
      rob.right_encoder?.toFixed(0),
      rob.team || 'red',
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

function sendTeam() {
  postJson('/team', {
    robot_id: document.getElementById('ov-id').value,
    team: document.getElementById('ov-team').value,
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

function updateTrainingInfo(state) {
  const infoEl = document.getElementById('training-info');
  if (state.run_number !== undefined && state.sim_time) {
    const elapsed = (state.sim_time - state.sim_start).toFixed(1);
    const r = (state.run_record || {}).red || 0;
    const b = (state.run_record || {}).blue || 0;
    infoEl.textContent = `Run: ${state.run_number}, R: ${r}, B: ${b}, Time: ${elapsed}s`;
  }
}

window.addEventListener('resize', render);

function onWsMessage(event) {
  const msg = JSON.parse(event.data);
  const handlers = {
    init: () => {
      robots = msg.robots;
      ball = msg.ball ?? ball;
      rebuildJoysticks();
    },
    ball: () => {
      ball = msg.data;
    },
    estimated_pose: () => {
      estimatedPoses[msg.data.robot_id] = msg.data;
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
    training_state: () => {
      updateTrainingInfo(msg.data);
    },
    telemetry: () => {
      robots[msg.data.robot_id] = { ...robots[msg.data.robot_id], ...msg.data };
    },
    virtual_robots: () => {
      for (const [id, data] of Object.entries(msg.data)) robots[id] = { ...robots[id], ...data };
    },
  };
  handlers[msg.type]?.();
  render();
  updateBallInfo();
  updateTable();
}

connectWebSocket();
render();
