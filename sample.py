import numpy as np
from sensing_network.pipeline import default_pipeline
from sensing_network.convert_utils import output_to_3dforce_json, output_to_json


def generate_random_network(n_nodes=10, n_links=20):

    def gen_links(nodes, n_links):
        links = []
        link_exist = {}
        while len(links) < n_links:
            s, t = np.random.choice(len(nodes), 2)
            s = int(s)
            t = int(t)
            if (s != t) and (not f'{s}-{t}' in link_exist):
                link_exist[f'{s}-{t}'] = True
                link_exist[f'{t}-{s}'] = True
                links.append([s, t])
        return links

    nodes = np.arange(n_nodes).astype(np.int16)
    links = gen_links(nodes, n_links)
    while len(np.unique(links)) < len(nodes):
        links = gen_links(n_links)

    links = [[s, t] if s < t else [t, s] for s, t in links]
    links.sort()

    return {'nodes': nodes, 'links': links}


sample_networks = {
    'simple': {
        'nodes': [0, 1, 2, 3],
        'links': [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    },
    'random': generate_random_network(n_nodes=10, n_links=20),
    'intersected': {
        # one of the mininum examples to show link intersection
        'nodes': [0, 1, 2, 3, 4, 5],
        'links': [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4],
                  [2, 3], [2, 4], [3, 4], [0, 5], [1, 5], [2, 5], [3, 5]]
    },
    'usage_scenario': {
        # network inspired by https://dshizuka.github.io/networkanalysis/networktypes_spatialnets.html
        'nodes':
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        'links': [[0, 3], [0, 5], [0, 7], [0, 8], [1, 10], [1, 11], [1, 14],
                  [1, 19], [2, 8], [2, 9], [2, 13], [2, 16], [3, 5], [3, 7],
                  [3, 15], [3, 16], [4, 6], [4, 11], [4, 12], [5, 17], [6, 9],
                  [6, 11], [6, 12], [6, 14], [7, 9], [7, 15], [8, 9], [8, 16],
                  [8, 17], [9, 18], [10, 11], [10, 14], [11, 12], [12, 14],
                  [12, 19], [13, 16], [13, 17], [14, 19], [15, 18], [16, 17]]
    }
}

if __name__ == '__main__':
    network = sample_networks['simple']
    nodes = network['nodes']
    links = network['links']
    links.sort()

    result = default_pipeline(nodes=nodes,
                              links=links,
                              node_radius=10.0,
                              link_radius=3.9,
                              mean_link_length=50,
                              resistance_range=[20.0e3, 300.0e3],
                              resistor_path_generation_kwargs={
                                  'vertical_step': 1.0,
                                  'path_margin': 0.65,
                                  'vertical_box_width': 1.2,
                                  'vertical_box_additional_height': 0.5,
                                  'min_node_link_overlap': 3.0,
                                  'vertical_resistivity': 930.0,
                                  'horizontal_resistivity': 570.0
                              },
                              use_out_node=False)

    output_to_3dforce_json(nodes,
                           links,
                           node_positions=result['node_positions'],
                           node_radius=result['node_radius'],
                           link_radius=result['link_radius'],
                           outfile_path='./network_renderer/data/nw.json')

    output_to_json(outfile_path='./result/cad_data.json', **result)

    import threading
    import webbrowser
    from http.server import HTTPServer, SimpleHTTPRequestHandler

    http_server = HTTPServer(('localhost', 8000), SimpleHTTPRequestHandler)
    threading.Thread(target=http_server.serve_forever, daemon=True).start()

    webbrowser.open('http://localhost:8000/network_renderer/')
