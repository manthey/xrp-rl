let latest = 0;
let cachedKey = null;
let cachedData = null;
let cachedInfo = null;

function fieldToCanvas(config, scale, gd, x, y) {
  return [(x + config.field_length_mm / 2) * scale + 1 + gd, (config.field_width_mm / 2 - y) * scale + 1];
}

function processQData(data, qvIndex) {
  const averages = {};
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
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
    maxH = Math.max(maxH, h);
    maxP1 = Math.max(maxP1, p1);
    const subKey = `${x},${y},${h},${p1}`;
    if (!sums[subKey]) sums[subKey] = { q: data.q[key].map(() => 0), c: 0 };
    const qVals = data.q[key];
    for (let i = 0; i < qVals.length; i++) {
      sums[subKey].q[i] += qVals[i];
      // to do max instead of average
      // sums[subKey].q[i] = Math.max(sums[subKey].q[i], qVals[i]);
    }
    sums[subKey].c++;
  }

  let minMaxVal, maxMaxVal;
  for (const key in sums) {
    const avg = sums[key].q.map((v) => v / sums[key].c);
    // for max instead of average
    // const avg = sums[key].q.map((v) => v);
    let maxIdx = 0;
    let maxVal = -Infinity;
    for (let i = 0; i < avg.length; i++) {
      if (avg[i] > maxVal) {
        maxVal = avg[i];
        maxIdx = i;
      }
    }
    if (minMaxVal === undefined) minMaxVal = maxMaxVal = maxVal;
    minMaxVal = Math.min(minMaxVal, maxVal);
    maxMaxVal = Math.max(maxMaxVal, maxVal);
    averages[key] = [maxIdx, maxVal];
  }

  return { averages, maxX, maxY, maxH, maxP1, minMaxVal, maxMaxVal };
}

function clamp(v, lo, hi) {
  return v < lo ? lo : v > hi ? hi : v;
}

function clickedToXY(msg, info) {
  const px = msg.clickX;
  const py = msg.clickY;
  const wx = (px - 1 - msg.gd) / msg.scale - msg.config.field_length_mm / 2;
  const wy = msg.config.field_width_mm / 2 - (py - 1) / msg.scale;
  const binW = msg.config.field_length_mm / (info.maxX + 1);
  const binH = msg.config.field_width_mm / (info.maxY + 1);
  const xIndex = clamp(Math.round((wx + msg.config.field_length_mm / 2) / binW - 0.5), 0, info.maxX);
  const yIndex = clamp(Math.round((wy + msg.config.field_width_mm / 2) / binH - 0.5), 0, info.maxY);
  return [xIndex, yIndex];
}

async function renderQState(msg) {
  latest = msg.queryNum;
  const rosetteMode = msg.type === 'render_rosette';

  const cacheKey = `${msg.qIndex}|${msg.qvIndex}`;
  if (cacheKey !== cachedKey) {
    const res = await fetch(`/q_files/${msg.qIndex}`);
    const data = await res.json();
    if (msg.queryNum !== latest) return;
    cachedData = data;
    cachedInfo = processQData(data, msg.qvIndex);
    cachedKey = cacheKey;
  }

  const data = cachedData;
  const info = cachedInfo;
  if (!data || !info) return;
  if (msg.queryNum !== latest) return;

  const canvas = new OffscreenCanvas(msg.width, msg.height);
  const ctx = canvas.getContext('2d');

  const range = info.maxMaxVal - info.minMaxVal || 1;
  const binW = rosetteMode ? msg.config.field_length_mm : msg.config.field_length_mm / (info.maxX + 1);
  const binH = rosetteMode ? msg.config.field_width_mm : msg.config.field_width_mm / (info.maxY + 1);
  const maxR = Math.min(binW, binH) * 0.48;
  const innerR = maxR * 0.2;
  const ringW = (maxR - innerR) / (info.maxP1 + 1);
  const dTheta = (2 * Math.PI) / (info.maxH + 1);

  const [xSel, ySel] = rosetteMode ? clickedToXY(msg, info) : [0, 0];
  const center = rosetteMode ? fieldToCanvas(msg.config, msg.scale, msg.gd, 0, 0) : null;

  const fontSize = Math.max(10, Math.round(msg.scale * 4));
  const needText = rosetteMode;
  if (needText) {
    ctx.font = `${fontSize}px monospace`;
    ctx.fillStyle = '#fff';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
  }

  const xStart = rosetteMode ? xSel : 0;
  const xEnd = rosetteMode ? xSel : info.maxX;
  const yStart = rosetteMode ? ySel : 0;
  const yEnd = rosetteMode ? ySel : info.maxY;

  for (let x = xStart; x <= xEnd; x++) {
    for (let y = yStart; y <= yEnd; y++) {
      const [cx, cy] = rosetteMode
        ? center
        : fieldToCanvas(
            msg.config,
            msg.scale,
            msg.gd,
            (x + 0.5) * binW - msg.config.field_length_mm / 2,
            (y + 0.5) * binH - msg.config.field_width_mm / 2,
          );

      for (let h = 0; h <= info.maxH; h++) {
        const theta = h * dTheta;
        const ca = -theta;

        for (let p1 = 0; p1 <= info.maxP1; p1++) {
          const subKey = `${x},${y},${h},${p1}`;
          const avg = info.averages[subKey];
          if (!avg) continue;

          const [actionIdx, maxVal] = avg;
          const action = data.actions[actionIdx];
          const r1 = msg.scale * (innerR + p1 * ringW);
          const r2 = msg.scale * (innerR + (p1 + 1) * ringW);
          const rel = ((maxVal - info.minMaxVal) / range) * 0.6 + 0.2;

          ctx.beginPath();
          ctx.arc(cx, cy, r2, ca - dTheta / 2, ca + dTheta / 2);
          ctx.arc(cx, cy, r1, ca + dTheta / 2, ca - dTheta / 2, true);
          ctx.fillStyle = `rgba(50,50,50,${rel})`;
          ctx.fill();

          const vx = action[1];
          const vy = action[2];
          const mag = Math.sqrt(vx * vx + vy * vy);
          if (!mag) continue;

          const rMid = ((2 / 3) * (r2 ** 3 - r1 ** 3)) / (r2 ** 2 - r1 ** 2);
          const arrowLen = Math.max(2, Math.min(msg.scale * ringW, rMid * dTheta) * 0.9) * mag;
          const ux = vx / mag;
          const uy = vy / mag;
          const dx = (ux * Math.cos(ca) - uy * Math.sin(ca)) * arrowLen;
          const dy = (ux * Math.sin(ca) + uy * Math.cos(ca)) * arrowLen;
          const mx = cx + Math.cos(ca) * rMid;
          const my = cy + Math.sin(ca) * rMid;
          const bx = mx - dx / 6;
          const by = my - dy / 6;
          const nx = -dy / 3;
          const ny = dx / 3;

          ctx.beginPath();
          ctx.moveTo(mx - dx / 2, my - dy / 2);
          ctx.lineTo(mx, my);
          ctx.strokeStyle = '#fff';
          ctx.lineWidth = arrowLen / 8;
          ctx.stroke();

          ctx.beginPath();
          ctx.moveTo(mx + dx / 2, my + dy / 2);
          ctx.lineTo(bx + nx, by + ny);
          ctx.lineTo(bx - nx, by - ny);
          ctx.closePath();
          ctx.fillStyle = '#fff';
          ctx.fill();

          if (needText) {
            const inv = arrowLen ? 1 / arrowLen : 0;
            const ox = -dy * inv;
            const oy = dx * inv;
            ctx.font = `${fontSize * 2.5}px sanserif`;
            ctx.fillStyle = '#0f0';
            ctx.textAlign = 'left';
            ctx.fillText(maxVal.toFixed(2), mx + ox * fontSize, my + oy * fontSize);
          }
        }
      }
    }
  }

  const bitmap = canvas.transferToImageBitmap();
  postMessage({ queryNum: msg.queryNum, bitmap }, [bitmap]);
}

onmessage = (e) => {
  if (e.data.type === 'clear') latest = e.data.queryNum;
  else renderQState(e.data);
};
