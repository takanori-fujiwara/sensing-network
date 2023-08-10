/*
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License
https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

Copyright (c) 2023, Takanori Fujiwara and S. Sandra Bae
All rights reserved.
*/


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

  // cut first cutThres
  const cutThres = 1500 // in ms
  config = {};

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
  for (const node in dataByNode) {
    let lastTime = 0;
    if (node != -1) {
      lastTime = dataByNode[node - 1].times.slice(-1)[0];
    }

    let nVals = 0;
    config[node] = 0.0;
    for (let i = 0; i < dataByNode[node].times.length; ++i) {
      const time = dataByNode[node].times[i];
      if (time >= lastTime + cutThres) {
        config[node] += dataByNode[node].values[i];
        nVals++;
      }
    }
    config[node] /= nVals;
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

socket.on("data", data => {
  const capValue = data.capValue;
  const connectType = data.connectType;
  // use smaller buffer sizes when not using a serial port
  const bufferSizeDiv = connectType == 'serial' ? 1 : 2;

  if (mode == "collection") {
    const time = performance.now() - startTime; // in millisec
    collectingData.push({
      "node": currentCollectingNode,
      "time": time,
      "value": capValue
    });
  } else {
    bufferedCapValues.push(parseFloat(capValue));
    if (bufferedCapValues.length > bufferSizeForCapValues / bufferSizeDiv) {
      bufferedCapValues.shift()
      const aveCapValue = bufferedCapValues.reduce((cum, val) => cum + val) / (bufferSizeForCapValues / bufferSizeDiv);
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

        document.querySelector("#selectedNode").innerHTML = `Node ${selectedNode}`;
      }
    }
  }
});