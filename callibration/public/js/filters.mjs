export function estimateMean(times, values, startTime) {
  let n = 0;
  let sum = 0.0;

  for (let i = 0; i < times.length; i++) {
    if (times[i] >= startTime) {
      sum += values[i];
      n++;
    }
  }

  return sum / n;
}

export function applyIIRLowPass(currentValue, newValue, dt, tau) {
  const alpha = Math.exp(-dt / tau);
  return alpha * currentValue + (1 - alpha) * newValue;
}

export function estimateIIRLowPass(times, values, startTime, tau) {
  let y = null;
  let lastTime = null;

  for (let i = 0; i < times.length; i++) {
    const t = times[i];
    if (t < startTime) continue;

    const x = values[i];

    if (y === null) {
      y = x;
      lastTime = t;
      continue;
    }

    const dt = t - lastTime;
    y = applyIIRLowPass(y, x, dt, tau);
    lastTime = t;
  }

  return y;
}

export function applyKalman(currentEstimate, P, measurement, R) {
  const K = P / (P + R);
  const newEstimate = currentEstimate + K * (measurement - currentEstimate);
  const newP = (1 - K) * P;
  return { newEstimate, newP };
}

export function estimateKalman(times, values, startTime, R, P0 = 1.0) {
  let x = null;
  let P = P0;

  for (let i = 0; i < times.length; i++) {
    const t = times[i];
    if (t < startTime) continue;

    const z = values[i];

    if (x === null) {
      x = z;
      continue;
    }

    ({newEstimate: x, newP: P} = applyKalman(x, P, z, R));
  }

  return x;
}