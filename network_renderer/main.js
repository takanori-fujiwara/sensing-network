// For more options, check: https://github.com/vasturiano/3d-force-graph/
export const draw = (data, {
  targetDom = '3d-network',
  backgroundColor = '#eeeeee',
  cameraPosition = [
    [],
    [], 0
  ],
  nodeRelSize = 10,
  nodeColor = () => '#444444',
  nodeOpacity = 1.0,
  nodeResolution = 32,
  linkWidth = 10,
  linkColor = () => '#966F33',
  linkOpacity = 1.0,
  linkResolution = 32
} = {}) =>
  ForceGraph3D()
    (document.getElementById(targetDom))
    .graphData(data)
    .cameraPosition(...cameraPosition)
    .backgroundColor(backgroundColor)
    .nodeRelSize(nodeRelSize) // basically, radius not diameter
    .nodeColor(nodeColor)
    .nodeOpacity(nodeOpacity)
    .nodeResolution(nodeResolution)
    .nodeLabel('id')
    .linkWidth(linkWidth) // basically, diameter
    .linkColor(linkColor)
    .linkOpacity(linkOpacity)
    .linkResolution(linkResolution);

const data = await d3.json('./data/nw.json', d3.autoType);

const mins = {
  'fx': 1e10,
  'fy': 1e10,
  'fz': 1e10
};
const maxs = {
  'fx': -1e10,
  'fy': -1e10,
  'fz': -1e10
};

for (const fi of ['fx', 'fy', 'fz']) {
  mins[fi] = data.nodes.reduce((acc, node) => node[fi] < acc ? node[fi] : acc, mins[fi]);
  maxs[fi] = data.nodes.reduce((acc, node) => node[fi] > acc ? node[fi] : acc, maxs[fi]);
}

const cameraPosition = [{
  x: (mins.fx + maxs.fx) * 0.5,
  y: (mins.fy + maxs.fy) * 0.5,
  z: maxs.fz * 5.0
}, {
  x: (mins.fx + maxs.fx) * 0.5,
  y: (mins.fy + maxs.fy) * 0.5,
  z: (mins.fz + maxs.fz) * 0.5,
},
  0
]

draw(data, {
  nodeRelSize: data.nodeRadius,
  linkWidth: data.linkRadius * 2,
  cameraPosition: cameraPosition
});