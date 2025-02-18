########## INIT ####################################################################################

##### Imports #####

### Standard ###
from copy import deepcopy
from random import randrange

### Special ###
import numpy as np

### Local ###
from ..env_config import env_var
from ..symbolic.utils import match_name, normalize_dist
from ..symbolic.symbols import ( ObjPose, GraspObj, extract_pose_as_homog, euclidean_distance_between_symbols )
from utils import ( snap_z_to_nearest_block_unit_above_zero, LogPickler, zip_dict_sorted_by_decreasing_value, 
                    deep_copy_memory_list, closest_ray_points )
from Bayes import BayesMemory


########## MEMORY ##################################################################################




class StateMemory:
    """ Contains memories of all the beliefs over time """

    def reset_memory( self ):
        """ Erase memory components """
        self.scan : list[GraspObj]   = list()
        self.mult : bool             = False
        self.bMem : BayesMemory      = BayesMemory()
        self.symH : Dict[uuid4,ThinSymbol] = dict()

    def __init__( self ):