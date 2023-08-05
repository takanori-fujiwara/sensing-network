## A Computational Design Pipeline to Fabricate Sensing Network Physicalizations

About
-----

* Source code to generate a sensing network physicalization introduced in **A Computational Design Pipeline to Fabricate Sensing Network Physicalizations**, S. Sandra Bae, Takanori Fujiwara, Anders Ynnerman, Ellen Yi-Luen Do, Michael L. Rivera, and Danielle Albers Szafir.
IEEE Transactions on Visualization and Computer Graphics and IEEE VIS 2023, forthcoming.

* Demonstration video: will be released soon.

******

Content
-----
* sensing_network: Python modules of the computational pipeline to design a sensing network physicalization.
* arduino: Source code to communicate with Arduino Uno R4 (both wired and WiFi connection examples).
* network_renderer: Web-based renderer to review a 3D network.
* calibration: Web-based interface for calibration and sensing test. 
* evaluation: Source code to regenerate the computational evaluation results shown in the paper.

******

Installation 
-----

### Requirements
* Python3 (latest)

* For `arduino`:
  * Arduino Uno R4 (Uno R3 is available after minor changes)
    - Wired connection: Either Arduino Uno R4 WiFi or Minima
    - WiFi connection: Arduino Uno R4 WiFi
  * Arduino IDE 2 (latest)

* For `network_renderer`:
  * Browser supporting JavaScript ES2015(ES6) and WebGL 2.0
  * Internet connection (to access D3 and 3d-force-graph library)

* For `calibration`: 
  * Browser supporting JavaScript ES2015(ES6)
  * Node.js (latest)
  * Internet connection

* Note: Tested with macOS Ventura, Arduino Uno R4 WiFi, and Google Chrome.

### Setup

* Download/Clone this repository

* Move to the downloaded repository, then:

    `pip3 install .` or `python3 -m pip install .`


******

Usage of Library
-----

* Import installed modules from python (e.g., `from sensing_network.pipeline import default_pipeline`). See `sample.py` for examples.
  - To run sample.py from the command line, use -i option (to review a 3D network with `network_renderer`):
    `python3 -i sample.py`

* Detailed documentations will be released soon.


******

How to build a sensing network
-----

(We are planning to provide an instruction video)

1. Use the computational pipeline as in sample.py to generate the information required to create a sensing network:

        'nodes': Node IDs.
        'links': Source and target node IDs for each link. 
        'node_positions': 3D node positions corresponding to 'nodes' in order.
        'node_radius': Node radius (i.e., sphere radius).
        'link_radius': Link radius (i.e., cylinder base radius).
        'resistor_links': Links selected as resistors.
        'resistances': Resistances of 'resistor_links' in order.
        'in_node': Node that should be connected to an external resistor connected to a Send pin.
        'out_node': Node that should be connected to a Receive pin (optional to use).
        'resistors_h_paths': Horizontal paths (xy-plane serpentine trace patterns) for the resistor embedding.
        'resistors_v_boxes': Vertical boxes (z-direction thick lines) for the resistor embedding.

2. From the above information, generate STL files for 3D printing.
  - output_to_stl (see sample.py) can convert the above information to a set of STL files
    - *.node.stl: STL files for nodes (for conductive material)
    - *.link.stl: STL files for links (for non-conductive material)
    - *.resistor.stl: STL files for resistors (for conductive material)
  - Or you can use output_to_json (see sample.py) to output the above information as a json file and use it to produce STL files with any CAD software (e.g., Rhino CAD with Grasshopper)
    - For the networks shown in our paper, we used Rhino CAD to make STL files more suitabel for 3D printing (e.g., making slight blank spaces around the boundaries of link and resistor objects).
    - We are planning to provide our grasshopper script.

3. Print a network using conductive (for nodes and resistor embedding) and non-conductive materials.
  - For example, you can use [PrusaSlicer](https://github.com/prusa3d/PrusaSlicer/) for this preparation.
    - Node objects (*.node.stl): conductive materials, any infill density (e.g., 20%)
    - Link objects (*.link.stl): non-conductive materials, high infill density (e.g., 90%)
    - Resistor objects (*.resistor.stl): conductive materials, very high infill density (e.g., 100%)

4. Build an electric circuit with Arduino Uno R4
  - Connect a resistor with large resistence (e.g., 1M ohm) to a Send pin of the Arduino and 'in_node'
  - If using 'out_node', connect 'out_node' to Receive pin. Otherwise, connect 'in_node' to a Receive pin.
  - Connect the Arduino to a power source

5. The electric circuit above shows a different voltage time delay at the Receive pin based on a touched node.
  - To measure the delays, you can use souce code in `arudino` directory.
    - Installation to Arduino: Upload `wired_connection.ino` or `wifi_connection.ino` to the Arduino via Arduino IDE
  - For calibration among time delays and touched nodes, you can refer to souce code in `calibration` directory.
    - Installation: move to `calibration` directory and then `npm install` in terminal.
    - Launching: `npm start` and then access to the indicated ip address from a Web browser
  - Note: Capacitive sensing is sensitive to various external conditions, such as human capacitance and environments (materials of a table where a network is placed). How to mitigate or effectively deal with this problem is remaining as future work.   

******
License
-----

See License.txt (Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License. Copyright: Takanori Fujiwara and S. Sandra Bae.)

******
How to cite
-----

S. Sandra Bae, Takanori Fujiwara, Anders Ynnerman, Ellen Yi-Luen Do, Michael L. Rivera, and Danielle Albers Szafir, "A Computational Design Pipeline to Fabricate Sensing Network Physicalizations." IEEE Transactions on Visualization and Computer Graphics, forthcoming.

