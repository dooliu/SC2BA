from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sc2ba.env.starcraft2.maps import sc2ba_maps


def get_map_params(map_name):
    map_param_registry = sc2ba_maps.get_smac_map_registry()
    return map_param_registry[map_name]
