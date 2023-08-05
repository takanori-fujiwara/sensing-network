from distutils.core import setup

setup(name='sensing_network',
      version=0.7,
      package_dir={'': '.'},
      install_requires=[
          'numpy',
          'scipy',
          'scikit-learn',
          'torch',
          'pytorch-lightning',
          'networkx',
          'pathos',
          'lcapy',
          'sympy',
          'pyvista',
      ],
      py_modules=[
          'sensing_network.convert_utils',
          'sensing_network.network_layout',
          'sensing_network.layout_adjustment',
          'sensing_network.resistance_optimization',
          'sensing_network.resistor_link_selection',
          'sensing_network.resistor_path_generation',
          'sensing_network.pipeline',
      ])
