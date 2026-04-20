# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fastapi",
#   "uvicorn",
#   "pydantic",
# ]
# ///

import argparse
import json
import math
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

app = FastAPI()
robots: dict[str, dict] = {}
websocket_clients: list[WebSocket] = []

FIELD_LENGTH_MM = 2500
FIELD_WIDTH_MM = 1500
GOAL_WIDTH_MM = 500
GOAL_DEPTH_MM = 120
CORNER_RADIUS_MM = 250
ROBOT_SIZE_MM = 200
BALL_DIAMETER_MM = 100
TAPE_WIDTH_MM = 25

TAPE_LINES = [
    {"x_mm": -FIELD_LENGTH_MM / 4,     "color": "blue"},
    {"x_mm":  FIELD_LENGTH_MM / 4,     "color": "red"},
    {"x_mm":  0,                        "color": "white"},
    {"x_mm": -FIELD_LENGTH_MM / 2 + 1, "color": "blue"},
    {"x_mm":  FIELD_LENGTH_MM / 2 - 1, "color": "red"},
]

FIELD_CONFIG = {
    "field_length_mm": FIELD_LENGTH_MM,
    "field_width_mm":  FIELD_WIDTH_MM,
    "goal_width_mm":   GOAL_WIDTH_MM,
    "goal_depth_mm":   GOAL_DEPTH_MM,
    "corner_radius_mm": CORNER_RADIUS_MM,
    "robot_size_mm":   ROBOT_SIZE_MM,
    "ball_diameter_mm": BALL_DIAMETER_MM,
    "tape_width_mm":   TAPE_WIDTH_MM,
    "tape_lines":      TAPE_LINES,
}


class Telemetry(BaseModel):
    robot_id: str
    left_encoder: float
    right_encoder: float
    distance_cm: float
    reflectance_left: float
    reflectance_right: float
    cmd_vel_left: float
    cmd_vel_right: float
    world_x_mm: float | None = None
    world_y_mm: float | None = None
    world_heading_deg: float | None = None
    pose_accuracy: float | None = None


class BallState(BaseModel):
    world_x_mm: float
    world_y_mm: float


class PoseOverride(BaseModel):
    robot_id: str
    world_x_mm: float
    world_y_mm: float
    world_heading_deg: float


ball_state: dict = {"world_x_mm": 0.0, "world_y_mm": 0.0}


@app.post("/telemetry")
async def receive_telemetry(telemetry: Telemetry):
    data = telemetry.model_dump()
    if telemetry.robot_id not in robots:
        robots[telemetry.robot_id] = {}
    robots[telemetry.robot_id].update(data)
    await broadcast({"type": "telemetry", "data": data})
    return {"status": "ok"}


@app.post("/ball")
async def update_ball(state: BallState):
    ball_state.update(state.model_dump())
    await broadcast({"type": "ball", "data": ball_state})
    return {"status": "ok"}


@app.post("/pose_override")
async def override_pose(override: PoseOverride):
    robot_id = override.robot_id
    if robot_id not in robots:
        robots[robot_id] = {"robot_id": robot_id}
    robots[robot_id]["world_x_mm"] = override.world_x_mm
    robots[robot_id]["world_y_mm"] = override.world_y_mm
    robots[robot_id]["world_heading_deg"] = override.world_heading_deg
    await broadcast({"type": "pose_override", "data": override.model_dump()})
    return {"status": "ok"}


@app.get("/robots")
async def get_robots():
    return robots


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    websocket_clients.append(ws)
    try:
        await ws.send_json({"type": "init", "robots": robots, "ball": ball_state})
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        websocket_clients.remove(ws)


async def broadcast(message: dict):
    disconnected = []
    for ws in websocket_clients:
        try:
            await ws.send_json(message)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        websocket_clients.remove(ws)


def build_html(config: dict) -> str:
    config_json = json.dumps(config)
    return r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>XRP Soccer Field</title>
<style>
  * { box-sizing: border-box; }
  body {
    background: #1a1a1a;
    color: #e0e0e0;
    font-family: monospace;
    margin: 0;
    padding: 0;
  }
  #canvas-container {
    width: 100vw;
    overflow: hidden;
  }
  canvas {
    display: block;
    background: #2d6a2d;
    width: 100%;
    height: auto;
  }
  #panel {
    padding: 10px;
  }
  table { border-collapse: collapse; width: 100%; font-size: 11px; }
  td, th { border: 1px solid #444; padding: 3px 6px; }
  th { background: #333; }
  .section-title { margin: 10px 0 4px; font-weight: bold; font-size: 13px; }
  input[type=number] { width: 70px; background: #333; color: #e0e0e0; border: 1px solid #666; }
  input[type=text]   { background: #333; color: #e0e0e0; border: 1px solid #666; }
  button { background: #555; color: #e0e0e0; border: 1px solid #888; cursor: pointer; padding: 2px 8px; }
  button:hover { background: #777; }
</style>
</head>
<body>
<div id="canvas-container">
  <canvas id="field"></canvas>
</div>
<div id="panel">
  <div class="section-title">Robots</div>
  <table id="robot-table">
    <thead><tr>
      <th>ID</th><th>X mm</th><th>Y mm</th><th>Hdg</th>
      <th>Dist cm</th><th>Refl L</th><th>Refl R</th>
      <th>Enc L</th><th>Enc R</th>
    </tr></thead>
    <tbody id="robot-tbody"></tbody>
  </table>
  <div class="section-title">Pose Override</div>
  <div>
    Robot ID: <input id="ov-id" type="text" style="width:80px">
    X mm: <input id="ov-x" type="number" value="0">
    Y mm: <input id="ov-y" type="number" value="0">
    Hdg: <input id="ov-hdg" type="number" value="0">
    <button onclick="sendOverride()">Set</button>
  </div>
</div>
<script>
const C = """ + config_json + r""";

const canvas = document.getElementById('field');
const ctx = canvas.getContext('2d');

let scale = 1;
let gd = 0;

function computeLayout() {
  const availableWidth = window.innerWidth;
  const goalDepthFraction = 2 * C.goal_depth_mm / C.field_length_mm;
  scale = availableWidth / (C.field_length_mm + 2 * C.goal_depth_mm + 2);
  gd = Math.round(scale * C.goal_depth_mm);
  canvas.width  = availableWidth;
  canvas.height = Math.round(scale * C.field_width_mm) + 2;
}

const s = mm => mm * scale;

function fieldToCanvas(x, y) {
  return [
    (x + C.field_length_mm / 2) * scale + 1 + gd,
    (C.field_width_mm / 2 - y) * scale + 1,
  ];
}

let robots = {};
let ball = {world_x_mm: 0, world_y_mm: 0};

function drawField() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.save();

  const fw   = s(C.field_length_mm);
  const fh   = s(C.field_width_mm);
  const r    = s(C.corner_radius_mm);
  const gw   = s(C.goal_width_mm);
  const ox   = 1 + gd;
  const oy   = 1;
  const gTop = oy + (fh - gw) / 2;

  for (const line of C.tape_lines) {
    const [lx] = fieldToCanvas(line.x_mm, 0);
    ctx.strokeStyle = line.color;
    ctx.lineWidth = Math.max(1, Math.round(s(C.tape_width_mm)));
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
    const wallX   = side === 1 ? ox + fw : ox;
    const wallDir = side === 1 ? 1 : -1;

    ctx.fillStyle = '#2d6a2d';
    ctx.fillRect(wallDir === 1 ? wallX : wallX - gd, gTop, gd, gw);

    ctx.strokeStyle = '#aaaaaa';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(wallX, gTop);
    ctx.lineTo(wallX + wallDir * gd, gTop);
    ctx.lineTo(wallX + wallDir * gd, gTop + gw);
    ctx.lineTo(wallX, gTop + gw);
    ctx.stroke();
  }

  ctx.restore();
}

function drawBall() {
  const [cx, cy] = fieldToCanvas(ball.world_x_mm, ball.world_y_mm);
  const r = s(C.ball_diameter_mm / 2);
  ctx.save();
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, 2 * Math.PI);
  ctx.fillStyle = '#f0f0f0';
  ctx.fill();
  ctx.strokeStyle = '#333333';
  ctx.lineWidth = 1;
  ctx.stroke();
  ctx.restore();
}

function drawRobots() {
  const rs = s(C.robot_size_mm / 2);
  for (const id in robots) {
    const rob = robots[id];
    const x   = rob.world_x_mm ?? null;
    const y   = rob.world_y_mm ?? null;
    if (x === null || y === null) continue;
    const hdg = rob.world_heading_deg ?? 0;
    const [cx, cy] = fieldToCanvas(x, y);
    const rad = hdg * Math.PI / 180;

    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(-rad);
    ctx.fillStyle = id.toLowerCase().includes('red')  ? '#cc3333' :
                    id.toLowerCase().includes('blue') ? '#3366cc' : '#888888';
    ctx.fillRect(-rs, -rs, rs * 2, rs * 2);
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1;
    ctx.strokeRect(-rs, -rs, rs * 2, rs * 2);
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, -rs, rs, rs * 0.3);
    ctx.restore();

    ctx.fillStyle = '#ffffff';
    ctx.font = `${Math.max(9, Math.round(s(40)))}px monospace`;
    ctx.fillText(id, cx + rs + 2, cy);
  }
}

function render() {
  computeLayout();
  drawField();
  drawBall();
  drawRobots();
}

function updateTable() {
  const tbody = document.getElementById('robot-tbody');
  tbody.innerHTML = '';
  for (const id in robots) {
    const rob = robots[id];
    const tr = document.createElement('tr');
    for (const f of [
      id,
      rob.world_x_mm?.toFixed(0)        ?? '-',
      rob.world_y_mm?.toFixed(0)        ?? '-',
      rob.world_heading_deg?.toFixed(1) ?? '-',
      rob.distance_cm?.toFixed(1)       ?? '-',
      rob.reflectance_left?.toFixed(3)  ?? '-',
      rob.reflectance_right?.toFixed(3) ?? '-',
      rob.left_encoder?.toFixed(0)      ?? '-',
      rob.right_encoder?.toFixed(0)     ?? '-',
    ]) {
      const td = document.createElement('td');
      td.textContent = f;
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
}

function sendOverride() {
  fetch('/pose_override', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      robot_id:          document.getElementById('ov-id').value,
      world_x_mm:        parseFloat(document.getElementById('ov-x').value),
      world_y_mm:        parseFloat(document.getElementById('ov-y').value),
      world_heading_deg: parseFloat(document.getElementById('ov-hdg').value),
    }),
  });
}

window.addEventListener('resize', render);

const ws = new WebSocket(`ws://${location.host}/ws`);
ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  if (msg.type === 'init') {
    robots = msg.robots;
    ball   = msg.ball ?? ball;
  } else if (msg.type === 'telemetry') {
    const d = msg.data;
    robots[d.robot_id] = {...(robots[d.robot_id] ?? {}), ...d};
  } else if (msg.type === 'ball') {
    ball = msg.data;
  } else if (msg.type === 'pose_override') {
    const d = msg.data;
    if (robots[d.robot_id]) {
      robots[d.robot_id].world_x_mm        = d.world_x_mm;
      robots[d.robot_id].world_y_mm        = d.world_y_mm;
      robots[d.robot_id].world_heading_deg = d.world_heading_deg;
    }
  }
  render();
  updateTable();
};

render();
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return build_html(FIELD_CONFIG)


def main():
    parser = argparse.ArgumentParser(description="XRP Soccer Field Telemetry Visualizer")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", "-p", type=int, default=8080)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
