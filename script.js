const BOUNDS = { fern: [-2.2, 2.7, 0, 10], sier: [0, 1, 0, Math.sqrt(3) / 2] };
const DEFAULT = [-2, 2, -2, 2];
const cv = document.getElementById("cv"),
  ctx = cv.getContext("2d");
const typeEl = document.getElementById("type"),
  itersEl = document.getElementById("iters"),
  info = document.getElementById("info");
let view,
  cw,
  ch,
  timer = null,
  raf = 0;

function fit() {
  // size the canvas to the free space, keep it square-ish
  const m = document.querySelector("main");
  cw = Math.max(
    200,
    Math.floor(Math.min(m.clientWidth - 16, m.clientHeight - 16, 1000)),
  );
  ch = cw;
  cv.width = cw;
  cv.height = ch;
}
function resetView() {
  const [x0, x1, y0, y1] = BOUNDS[typeEl.value] || DEFAULT;
  view = {
    cx: (x0 + x1) / 2,
    cy: (y0 + y1) / 2,
    u: Math.max((x1 - x0) / cw, (y1 - y0) / ch),
  };
}
function pal(t, out, k) {
  t = Math.min(t, 1) * 1.5;
  out[k] = 255 * (0.5 + 0.5 * Math.cos(6.2832 * (t + 0.0)));
  out[k + 1] = 255 * (0.5 + 0.5 * Math.cos(6.2832 * (t + 0.33)));
  out[k + 2] = 255 * (0.5 + 0.5 * Math.cos(6.2832 * (t + 0.67)));
  out[k + 3] = 255;
}
// Complex coordinates of pixel (i,j) on a w x h grid covering the full canvas.
function coords(w, h) {
  const sx = cw / w,
    sy = ch / h;
  return (i, j) => [
    view.cx + ((i + 0.5) * sx - cw / 2) * view.u,
    view.cy - ((j + 0.5) * sy - ch / 2) * view.u,
  ];
}

function escapeTime(kind, w, h, maxIter) {
  const at = coords(w, h),
    img = ctx.createImageData(w, h),
    d = img.data,
    tol2 = 1e-12;
  for (let j = 0; j < h; j++)
    for (let i = 0; i < w; i++) {
      const [cr, ci] = at(i, j);
      let n = 0,
        zr = 0,
        zi = 0;
      if (kind === "mandel") {
        const q = (cr - 0.25) * (cr - 0.25) + ci * ci;
        if (
          q * (q + (cr - 0.25)) <= 0.25 * ci * ci ||
          (cr + 1) * (cr + 1) + ci * ci <= 0.0625
        )
          n = 0;
        else
          for (let k = 1; k <= maxIter; k++) {
            const t = zr * zr - zi * zi + cr;
            zi = 2 * zr * zi + ci;
            zr = t;
            if (zr * zr + zi * zi > 4) {
              n = k;
              break;
            }
          }
      } else if (kind === "ship") {
        for (let k = 1; k <= maxIter; k++) {
          const x = Math.abs(zr),
            y = Math.abs(zi);
          zr = x * x - y * y + cr;
          zi = 2 * x * y + ci;
          if (zr * zr + zi * zi > 4) {
            n = k;
            break;
          }
        }
      } else {
        // newton / nova: count steps until the iteration settles
        n = maxIter;
        zr = kind === "nova" ? 1 : cr;
        zi = kind === "nova" ? 0 : ci;
        for (let k = 1; k <= maxIter; k++) {
          const a = zr * zr - zi * zi,
            b = 2 * zr * zi,
            dd = a * a + b * b;
          if (dd === 0) {
            n = k;
            break;
          }
          let nr, ni;
          if (kind === "newton") {
            nr = (2 * zr + a / dd) / 3;
            ni = (2 * zi - b / dd) / 3;
          } else {
            nr = zr - (zr - a / dd) / 3 + cr;
            ni = zi - (zi + b / dd) / 3 + ci;
          }
          const dr = nr - zr,
            di = ni - zi;
          zr = nr;
          zi = ni;
          if (dr * dr + di * di < tol2 || zr * zr + zi * zi > 1e12) {
            n = k;
            break;
          }
        }
      }
      const k4 = (j * w + i) * 4,
        dark = kind === "newton" || kind === "nova" ? n >= maxIter : n === 0;
      if (dark) {
        d[k4] = 12;
        d[k4 + 1] = 10;
        d[k4 + 2] = 24;
        d[k4 + 3] = 255;
      } else pal(Math.sqrt(n / maxIter), d, k4);
    }
  return img;
}

function ifs(kind, w, h, thousands) {
  const [x0, x1, y0, y1] = [
    view.cx - (view.u * cw) / 2,
    view.cx + (view.u * cw) / 2,
    view.cy - (view.u * ch) / 2,
    view.cy + (view.u * ch) / 2,
  ];
  const target = thousands * 1000,
    N = 2000,
    counts = new Float32Array(w * h);
  const xs = new Float64Array(N),
    ys = new Float64Array(N),
    sq = Math.sqrt(3) / 2;
  let got = 0;
  for (let s = 0; s < 3000 && got < target; s++) {
    for (let c = 0; c < N; c++) {
      const x = xs[c],
        y = ys[c],
        r = Math.random();
      if (kind === "fern") {
        if (r < 0.01) {
          xs[c] = 0;
          ys[c] = 0.16 * y;
        } else if (r < 0.86) {
          xs[c] = 0.85 * x + 0.04 * y;
          ys[c] = -0.04 * x + 0.85 * y + 1.6;
        } else if (r < 0.93) {
          xs[c] = 0.2 * x - 0.26 * y;
          ys[c] = 0.23 * x + 0.22 * y + 1.6;
        } else {
          xs[c] = -0.15 * x + 0.28 * y;
          ys[c] = 0.26 * x + 0.24 * y + 0.44;
        }
      } else {
        const v = Math.floor(r * 3);
        xs[c] = (x + [0, 1, 0.5][v]) / 2;
        ys[c] = (y + [0, 0, sq][v]) / 2;
      }
      if (s < 20) continue;
      const px = xs[c],
        py = ys[c];
      if (px >= x0 && px < x1 && py >= y0 && py < y1) {
        const i = Math.min(w - 1, Math.floor(((px - x0) / (x1 - x0)) * w));
        const j = Math.min(h - 1, Math.floor(((y1 - py) / (y1 - y0)) * h));
        counts[j * w + i]++;
        got++;
      }
    }
  }
  let max = 0;
  for (let k = 0; k < counts.length; k++) {
    counts[k] = Math.log1p(counts[k]);
    if (counts[k] > max) max = counts[k];
  }
  const img = ctx.createImageData(w, h),
    d = img.data;
  for (let k = 0; k < counts.length; k++) {
    if (counts[k] === 0) {
      d[k * 4] = 12;
      d[k * 4 + 1] = 10;
      d[k * 4 + 2] = 24;
      d[k * 4 + 3] = 255;
    } else pal(0.15 + (0.6 * counts[k]) / max, d, k * 4);
  }
  return img;
}

const off = document.createElement("canvas");
function draw(scale) {
  const t0 = performance.now(),
    w = Math.max(1, Math.floor(cw * scale)),
    h = Math.max(1, Math.floor(ch * scale));
  const kind = typeEl.value,
    n = Math.max(1, Math.min(2000, parseInt(itersEl.value) || 100));
  const img =
    kind === "fern" || kind === "sier"
      ? ifs(kind, w, h, n)
      : escapeTime(kind, w, h, n);
  off.width = w;
  off.height = h;
  off.getContext("2d").putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, 0, 0, cw, ch);
  const base = Math.max(
    ((BOUNDS[kind] || DEFAULT)[1] - (BOUNDS[kind] || DEFAULT)[0]) / cw,
    1e-300,
  );
  info.textContent =
    "Zoom x" +
    (base / view.u).toExponential(1).replace("e+", "e") +
    (scale === 1 ? "  |  " + Math.round(performance.now() - t0) + " ms" : "");
}
function interact() {
  // coarse preview now, sharp image once things go quiet
  if (!raf)
    raf = requestAnimationFrame(() => {
      raf = 0;
      draw(0.25);
    });
  clearTimeout(timer);
  timer = setTimeout(() => draw(1), 150);
}
function full() {
  clearTimeout(timer);
  draw(1);
}

cv.addEventListener(
  "wheel",
  (e) => {
    e.preventDefault();
    const r = cv.getBoundingClientRect(),
      mx = ((e.clientX - r.left) * cw) / r.width,
      my = ((e.clientY - r.top) * ch) / r.height;
    const f = e.deltaY < 0 ? 0.8 : 1.25,
      nu = Math.max(view.u * f, 1e-13);
    const x = view.cx + (mx - cw / 2) * view.u,
      y = view.cy - (my - ch / 2) * view.u;
    view.u = nu;
    view.cx = x - (mx - cw / 2) * nu;
    view.cy = y + (my - ch / 2) * nu;
    interact();
  },
  { passive: false },
);
let drag = null;
cv.addEventListener("pointerdown", (e) => {
  drag = { x: e.clientX, y: e.clientY };
  cv.setPointerCapture(e.pointerId);
});
cv.addEventListener("pointermove", (e) => {
  if (!drag) return;
  const k = cw / cv.getBoundingClientRect().width;
  view.cx -= (e.clientX - drag.x) * k * view.u;
  view.cy += (e.clientY - drag.y) * k * view.u;
  drag = { x: e.clientX, y: e.clientY };
  interact();
});
cv.addEventListener("pointerup", () => {
  drag = null;
});
cv.addEventListener("dblclick", () => {
  resetView();
  full();
});
document.getElementById("reset").onclick = () => {
  resetView();
  full();
};
typeEl.onchange = () => {
  resetView();
  full();
};
itersEl.onchange = full;
window.addEventListener("resize", () => {
  fit();
  resetView();
  full();
});
fit();
resetView();
full();
