import { estimateMean, estimateIIRLowPass, estimateKalman } from './filters.mjs';
import fs from 'fs';

const data = JSON.parse(fs.readFileSync('./public/data/data.json', 'utf8'));
processData(data);

function processData(data) {
  const dataByNode = {};
  for (const d of data) {
    const node = parseInt(d.node);
    if (!(node in dataByNode)) {
      dataByNode[node] = { 'times': [], 'values': [] };
    } else {
      dataByNode[node].times.push(parseFloat(d.time));
      dataByNode[node].values.push(parseFloat(d.value));
    }
  }
  
  const cutThres = 1500; // in ms
  const tau = 300;
  const R = 0.05;
  const config1 = {};
  const config2 = {};
  const config3 = {};

  for (const node in dataByNode) {
    let lastTime = 0;
    if (node != -1) {
      lastTime = dataByNode[node - 1].times.slice(-1)[0];
    }

    const startTime = lastTime + cutThres;
    const { times, values } = dataByNode[node];
    config1[node] = estimateMean(times, values, startTime);
    config2[node] = estimateIIRLowPass(times, values, startTime, tau);
    config3[node] = estimateKalman(times, values, startTime, R);
  }

  console.log("Average Configuration:", config1);
  console.log("IIR Low-Pass Configuration:", config2);
  console.log("Kalman Configuration:", config3);

    const vegaLiteSpec = {
    $schema: 'https://vega.github.io/schema/vega-lite/v5.json',
    description: 'Values from Arduino.',
    width: 800,
    data: {
      values: data
    },
    mark: 'line',
    encoding: {
      x: { field: "time", type: "temporal" },
      y: { field: "value", type: "quantitative" },
      color: { field: "node", type: "nominal" }
    }
  };

    const htmlContent = `<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
  <script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
  <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
  <style>
    body {
      font-family: Verdana, Geneva, Tahoma, sans-serif;
      margin: 1em;
    }
  </style>
</head>
<body>
  <h1>Arduino Sensor Data Visualization</h1>
  <div id="vis"></div>
  <script type="text/javascript">
    const spec = ${JSON.stringify(vegaLiteSpec, null, 2)};
    vegaEmbed('#vis', spec);
  </script>
</body>
</html>`;

  fs.writeFileSync('./visualization.html', htmlContent, 'utf8');
  console.log("Visualization saved to visualization.html");
}