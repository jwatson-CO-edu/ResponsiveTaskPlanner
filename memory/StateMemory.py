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


##### Constants #####

_REVERSE_QUERIES = {
    "bluBlock": {'query': "a photo of a blue block"  , 'abbrv': "blu", },
    "ylwBlock": {'query': "a photo of a yellow block", 'abbrv': "ylw", },
    "grnBlock": {'query': "a photo of a green block" , 'abbrv': "grn", },
}



########## HELPER FUNCTIONS ########################################################################

# def observation_to_readings( obs, xform = None, zOffset = 0.0 ):
def observation_to_readings( obs, xform = None ):
    """ Parse the Perception Process output struct """
    rtnBel = []
    if xform is None:
        xform = np.eye(4)

    if isinstance( obs, dict ):
        obs = list( obs.values() )

    for item in obs:
        dstrb = {}
        tScan = item['Time']

        # WARNING: CLASSES WITH A ZERO PRIOR WILL NOT ACCUMULATE EVIDENCE!

        if isinstance( item['Probability'], dict ):
            for nam, prb in item['Probability'].items():
                if prb > 0.0001:
                    dstrb[ match_name( nam ) ] = prb
                else:
                    dstrb[ match_name( nam ) ] = env_var("_CONFUSE_PROB")

            for nam in env_var("_BLOCK_NAMES"):
                if nam not in dstrb:
                    dstrb[ nam ] = env_var("_CONFUSE_PROB")
                
            dstrb = normalize_dist( dstrb )

        if len( item['Pose'] ) == 16:
            if xform is not None:
                objPose = xform.dot( np.array( item['Pose'] ).reshape( (4,4,) ) ) 
            else:
                objPose = np.array( item['Pose'] ).reshape( (4,4,) )
            
            # # HACK: SNAP TO NEAREST BLOCK UNIT && SNAP ABOVE TABLE
            # objPose[2,3] = snap_z_to_nearest_block_unit_above_zero( objPose[2,3] + zOffset )
        else:
            raise ValueError( f"`observation_to_readings`: BAD POSE FORMAT!\n{item['Pose']}" )
        
        # item['CPCD']

        # Create reading
        rtnObj = GraspObj( 
            labels = dstrb, 
            pose   = ObjPose( objPose ), 
            ts     = tScan, 
            count  = item['Count'], 
            score  = 0.0,
            cpcd   = item['CPCD'],
        )

        # Transform CPCD
        mov = xform.copy()
        # mov[2,3] += zOffset
        rtnObj.cpcd.transform( mov )

        # Store mask centroid ray
        rtnObj.meta['rayOrg'] = xform[0:3,3].reshape(3)
        rtnObj.meta['rayDir'] = np.dot( xform[0:3,0:3], item['camRay'].reshape( (3,1,) ) ).reshape(3)

        rtnBel.append( rtnObj )
    return rtnBel





########## STATE MEMORY ############################################################################

class StateMemory:
    """ Manages the current concept of state """


    def reset_memory( self ):
        """ Erase memory components """
        self.obsv : list[dict]     = list()
        self.scan : list[GraspObj] = list()
        self.bMem : BayesMemory    = BayesMemory()


    def __init__( self ):
        """ Get ready to track objects """
        self.reset_memory()


    def observe( self, percList, xform = None, zOffset = 0.0 ):
        """ Process observations from the perception stack """
        self.obsv = deepcopy( percList )
        self.scan = observation_to_readings( self.obsv, xform, zOffset )


    def update_beliefs( self ):
        """ Aggregate readings in a (more?) sane way """
        # ISSUE: THE CAMERA PLACES THE OBSERVATIONS FAR APART IN DIFFERENT VIEWS

        def cluster_distributions( readList : list[GraspObj] , N : int ):
            """ Attempt to group `distList` into `N` clusters """
            clstrDex = [randrange(N) for _ in range( len( readList ) )]
            vectrLst = list()
            for rdng in readList:
                rdng.labels




        # Group obervations by their distributions

        # Merge before snap
        # Snap before Bayes
        # Bayesian Update
        
