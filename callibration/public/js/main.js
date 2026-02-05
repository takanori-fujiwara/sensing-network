/*
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License
https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

Copyright (c) 2023, Takanori Fujiwara and S. Sandra Bae
All rights reserved.
*/

import { estimateMean, estimateIIRLowPass, estimateKalman, applyIIRLowPass, applyKalman } from './filters.mjs';

const socket = io();

let mode = null; // collection or demo

// for data collection and preprocessing
const collectingData = [];
let currentCollectingNode = -1;
let startTime = null;
let intervalId = null;

// for selection demo
let config = null;
const bufferSizeForCapValues = 10;
const bufferSizeForSelectedNodes = 10;
let prevSelectedNode = -1;
const bufferedCapValues = [];
const bufferedSelectedNodes = [];

let iirFilteredValue = null;
let kalmanEstimate = null;
let kalmanP = 1.0;
var lastTime = 0.0;

const demoData = [];
let demoStartTime = null;

// for filtering
const cutThres = 1500 // in ms
const tau = 125; // for IIR low-pass
const R = 0.01; // for Kalman

const processData = (data) => {
  document.querySelector("#currentCollectingNode").innerHTML = "Processing collected data";

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
  vegaEmbed("#vis", vegaLiteSpec);

  const dataByNode = {
  }
  for (const d of data) {
    const node = parseInt(d.node);
    if (!(node in dataByNode)) {
      dataByNode[node] = { 'times': [], 'values': [] };
    } else {
      dataByNode[node].times.push(parseFloat(d.time));
      dataByNode[node].values.push(parseFloat(d.value));
    }
  }

  config = {};

  for (const node in dataByNode) {
    let lastTime = 0;
    if (node != -1) {
      lastTime = dataByNode[node - 1].times.slice(-1)[0];
    }

    const startTime = lastTime + cutThres;
    const { times, values } = dataByNode[node];
    config[node] = estimateMean(times, values, startTime);
    // config[node] = estimateIIRLowPass(times, values, startTime, tau);
    // config[node] = estimateKalman(times, values, startTime, R);
  }

  socket.emit("endProcessing", JSON.stringify(config));
  document.querySelector("#currentCollectingNode").innerHTML = "Finished processing data (config data is in data directory)";
};

const selectCloseCapValNode = (aveCapValue, config) => {
  const nodes = Object.keys(config);
  const repCapValues = Object.values(config);

  let selectedNode = -1;
  let minDiff = 1e100;
  for (let i = 0; i < nodes.length; ++i) {
    const diff = Math.abs(repCapValues[i] - aveCapValue);
    if (diff < minDiff) {
      selectedNode = nodes[i]
      minDiff = diff;
    }
  }

  return parseInt(selectedNode);
}

document.querySelector("#collectionStartButton").addEventListener("click", () => {
  mode = "collection";

  socket.emit("io", {
    "statuses": "1"
  });
  startTime = performance.now();

  currentCollectingNode = -1;
  collectingData.length = 0;

  document.querySelector("#currentCollectingNode").innerHTML = 'Wait';

  intervalId = setInterval(() => {
    currentCollectingNode++;
    if (currentCollectingNode < document.querySelector("#nNodesInput").value) {
      document.querySelector("#currentCollectingNode").innerHTML =
        `Touch Node ${currentCollectingNode} (total ${document.querySelector("#nNodesInput").value} nodes)`;
    } else {
      document.querySelector("#currentCollectingNode").innerHTML = "Done (raw json data is in data directory)";
      socket.emit("endCollection", JSON.stringify(collectingData));
      clearInterval(intervalId);
      processData(collectingData);
    }
  }, 5000);
});

document.querySelector("#demoStartButton").addEventListener("click", () => {
  mode = "demo";
  demoData.length = 0;
  demoStartTime = performance.now();
  bufferedCapValues.length = 0;
  bufferedSelectedNodes.length = 0;
  prevSelectedNode = -1;

  iirFilteredValue = null;
  kalmanEstimate = null;
  kalmanP = 1.0;

  if (config) {
    socket.emit("io", {
      "statuses": "1"
    });
  } else {
    fetch("./data/config.json")
      .then(response => response.json())
      .then(json => {
        config = json;
        socket.emit("io", {
          "statuses": "1"
        });
      });
  }
});

document.querySelector("#demoSaveButton").addEventListener("click", () => {
  if (demoData.length > 0) {
    socket.emit("saveDemoData", JSON.stringify(demoData));
    document.querySelector("#selectedNode").innerHTML = `Demo data saved (${demoData.length} records)`;
  } else {
    document.querySelector("#selectedNode").innerHTML = "No demo data to save";
  }
});

socket.on("data", data => {
  const capValue = data.capValue;
  const connectType = data.connectType;
  // use smaller buffer sizes when not using a serial port
  const bufferSizeDiv = connectType == 'serial' ? 1 : 2;
  const dt = performance.now() - lastTime;
  lastTime = performance.now();

  if (mode == "collection") {
    const time = performance.now() - startTime; // in millisec
    collectingData.push({
      "node": currentCollectingNode,
      "time": time,
      "value": capValue
    });
  } else if (mode == "demo") {
    // IIR low-pass
    /*const filteredValue = applyIIRLowPass(bufferedCapValues.length > 0 ? bufferedCapValues[bufferedCapValues.length -1] : parseFloat(capValue), parseFloat(capValue), dt, tau);
    bufferedCapValues.push(filteredValue);
    if (bufferedCapValues.length > bufferSizeForCapValues / bufferSizeDiv) {
      const selectedNode = selectCloseCapValNode(filteredValue, config);
      document.querySelector("#selectedNode").innerHTML = `Node ${selectedNode}`;
    }*/
    // Kalman
    /*const {newEstimate, newP} = applyKalman(bufferedCapValues.length > 0 ? bufferedCapValues[bufferedCapValues.length -1] : parseFloat(capValue), newP, parseFloat(capValue), R);
    bufferedCapValues.push(newEstimate);
    if (bufferedCapValues.length > bufferSizeForCapValues / bufferSizeDiv) {
      const selectedNode = selectCloseCapValNode(newEstimate, config);
      document.querySelector("#selectedNode").innerHTML = `Node ${selectedNode}`;
    }*/
    const time = performance.now() - demoStartTime;
    const rawValue = parseFloat(capValue);

    const dataPoint = {
      "time": time,
      "rawValue": rawValue,
      "selectedNode": -1
    };

    if (iirFilteredValue === null) {
      iirFilteredValue = rawValue;
    } else {
      iirFilteredValue = applyIIRLowPass(iirFilteredValue, rawValue, dt, tau);
    }
    console.log(iirFilteredValue);
    dataPoint.iirFiltered = iirFilteredValue;

    if (kalmanEstimate === null) {
      kalmanEstimate = rawValue;
    } else {
      const { newEstimate, newP } = applyKalman(kalmanEstimate, kalmanP, rawValue, R);
      kalmanEstimate = newEstimate;
      kalmanP = newP;
    }
    console.log(kalmanEstimate);
    dataPoint.kalmanFiltered = kalmanEstimate;

    // Moving average
    bufferedCapValues.push(rawValue);
    if (bufferedCapValues.length > bufferSizeForCapValues / bufferSizeDiv) {
      bufferedCapValues.shift()
      const aveCapValue = bufferedCapValues.reduce((cum, val) => cum + val) / (bufferSizeForCapValues / bufferSizeDiv);
      dataPoint.movingAverageFiltered = aveCapValue;
      const tmpSelectedNode = selectCloseCapValNode(aveCapValue, config);
      bufferedSelectedNodes.push(tmpSelectedNode);

      if (bufferedSelectedNodes.length > bufferSizeForSelectedNodes) {
        bufferedSelectedNodes.shift();
        const counts = bufferedSelectedNodes.reduce((acc, node) => {
          if (node in acc) {
            acc[node]++;
          } else {
            acc[node] = 1;
          }
          return acc;
        }, {});

        const sortedNodesByCounts = Object.keys(counts).sort((a, b) => - counts[a] + counts[b]);
        let mostFreqNode = sortedNodesByCounts[0];
        let selectedNode = -1;

        if (counts[mostFreqNode] > bufferSizeForSelectedNodes * 0.8) {
          selectedNode = mostFreqNode;
          prevSelectedNode = selectedNode;
          bufferedCapValues.length = 0;
        } else {
          selectedNode = prevSelectedNode;
        }
        dataPoint.selectedNode = selectedNode;
        document.querySelector("#selectedNode").innerHTML = `Node ${selectedNode}`;
      } else {
        dataPoint.movingAverageFiltered = null; // Not enough data yet
      }
    }
    demoData.push(dataPoint);
  }
});