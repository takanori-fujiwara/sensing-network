import fs from 'fs';

const dataDir = './public/data/';
const files = fs.readdirSync(dataDir).filter(f => f.startsWith('demo_data_'));

const latestFile = files.sort().pop();
const data = JSON.parse(fs.readFileSync(dataDir + latestFile, 'utf8'));

processDemoData(data);

function processDemoData(data) {
  const comparisonDataIIR = [];
  const comparisonDataKalman = [];
  const comparisonDataMovingAvg = [];

  for (const d of data) {
    const time = d.time;
    const rawValue = d.rawValue;

    // IIR comparison
    comparisonDataIIR.push({
      time: time,
      value: rawValue,
      type: "Raw"
    });
    if (d.iirFiltered !== undefined) {
      comparisonDataIIR.push({
        time: time,
        value: d.iirFiltered,
        type: "IIR Filtered"
      });
    }

    // Kalman comparison
    comparisonDataKalman.push({
      time: time,
      value: rawValue,
      type: "Raw"
    });
    if (d.kalmanFiltered !== undefined) {
      comparisonDataKalman.push({
        time: time,
        value: d.kalmanFiltered,
        type: "Kalman Filtered"
      });
    }

    // Moving Average comparison
    comparisonDataMovingAvg.push({
      time: time,
      value: rawValue,
      type: "Raw"
    });
    if (d.movingAverageFiltered !== undefined) {
      comparisonDataMovingAvg.push({
        time: time,
        value: d.movingAverageFiltered,
        type: "Moving Avg Filtered"
      });
    }
  }

  const iirSpec = {
    $schema: 'https://vega.github.io/schema/vega-lite/v5.json',
    description: 'IIR Low-Pass Filter Comparison',
    width: 900,
    height: 400,
    data: {
      values: comparisonDataIIR
    },
    mark: 'line',
    encoding: {
      x: { field: "time", type: "quantitative", title: "Time (ms)" },
      y: { field: "value", type: "quantitative", title: "Value" },
      color: { field: "type", type: "nominal", scale: { domain: ["Raw", "IIR Filtered"], range: ["lightblue", "darkblue"] } }
    }
  };

  const kalmanSpec = {
    $schema: 'https://vega.github.io/schema/vega-lite/v5.json',
    description: 'Kalman Filter Comparison',
    width: 900,
    height: 400,
    data: {
      values: comparisonDataKalman
    },
    mark: 'line',
    encoding: {
      x: { field: "time", type: "quantitative", title: "Time (ms)" },
      y: { field: "value", type: "quantitative", title: "Value" },
      color: { field: "type", type: "nominal", scale: { domain: ["Raw", "Kalman Filtered"], range: ["lightcoral", "darkred"] } }
    }
  };

  const movingAvgSpec = {
    $schema: 'https://vega.github.io/schema/vega-lite/v5.json',
    description: 'Moving Average Comparison',
    width: 900,
    height: 400,
    data: {
      values: comparisonDataMovingAvg
    },
    mark: 'line',
    encoding: {
      x: { field: "time", type: "quantitative", title: "Time (ms)" },
      y: { field: "value", type: "quantitative", title: "Value" },
      color: { field: "type", type: "nominal", scale: { domain: ["Raw", "Moving Avg Filtered"], range: ["lightgreen", "darkgreen"] } }
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
    .chart-container {
      margin-bottom: 2em;
    }
    h2 {
      color: #333;
      margin-top: 1em;
    }
  </style>
</head>
<body>
  <h1>Demo Data - Filter Comparison</h1>
  
  <div class="chart-container">
    <h2>1. IIR Low-Pass Filter</h2>
    <div id="vis-iir"></div>
  </div>

  <div class="chart-container">
    <h2>2. Kalman Filter</h2>
    <div id="vis-kalman"></div>
  </div>

  <div class="chart-container">
    <h2>3. Moving Average</h2>
    <div id="vis-moving-avg"></div>
  </div>

  <script type="text/javascript">
    const iirSpec = ${JSON.stringify(iirSpec, null, 2)};
    const kalmanSpec = ${JSON.stringify(kalmanSpec, null, 2)};
    const movingAverageSpec = ${JSON.stringify(movingAvgSpec, null, 2)};
    
    vegaEmbed('#vis-iir', iirSpec);
    vegaEmbed('#vis-kalman', kalmanSpec);
    vegaEmbed('#vis-moving-avg', movingAverageSpec);
  </script>
</body>
</html>`;

  fs.writeFileSync('./demo_visualization.html', htmlContent, 'utf8');
  console.log("Demo visualization saved to demo_visualization.html");
}