########## INIT ####################################################################################

import pickle, os, time
now = time.time 
from math import isnan
from collections import deque
from datetime import datetime

import numpy as np
np.set_printoptions( precision = 4 )

from aspire.symbols import GraspObj
from aspire.env_config import env_var



########## HELPER FUNCTIONS ########################################################################

def zip_dict_sorted_by_decreasing_value( dct ):
    """ Return a list of (k,v) tuples sorted by decreasing value """
    keys = list()
    vals = list()
    for k, v in dct.items():
        keys.append(k)
        vals.append(v)
    return sorted( zip( keys, vals ), key=lambda x: x[1], reverse=1)



########## MEMORY FUNCTIONS ########################################################################

def copy_as_LKG( sym : GraspObj ):
    """ Make a copy of this belief for the Last-Known-Good collection """
    rtnObj = sym.copy()
    rtnObj.LKG = True
    return rtnObj


def copy_readings_as_LKG( readLst ):
    """ Return a list of readings intended for the Last-Known-Good collection """
    rtnLst = list()
    for r in readLst:
        rtnLst.append( copy_as_LKG( r ) )
    return rtnLst


def mark_readings_LKG( readLst : list[GraspObj], val : bool = True ):
    """ Return a list of readings intended for the Last-Known-Good collection """
    for r in readLst:
        r.LKG = val


def entropy_factor( probs ):
    """ Return a version of Shannon entropy scaled to [0,1] """
    if isinstance( probs, dict ):
        probs = list( probs.values() )
    tot = 0.0
    # N   = 0
    for p in probs:
        pPos = max( p, 0.00001 )
        tot -= pPos * np.log( pPos )
            # N   += 1
    return tot / np.log( len( probs ) )


def set_quality_score( obj : GraspObj ):
    """ Calc the score for this `GraspObj` """
    score_i = (1.0 - entropy_factor( obj.labels )) * obj.count
    if isnan( score_i ):
        print( f"\nWARN: Got a NaN score with count {obj.count} and distribution {obj.labels}\n" )
        score_i = 0.0
    obj.score = score_i


def snap_z_to_nearest_block_unit_above_zero( z : float ):
    """ SNAP TO NEAREST BLOCK UNIT && SNAP ABOVE TABLE """
    sHalf = (env_var("_BLOCK_SCALE")/2.0)
    zBump = sHalf + env_var("_Z_TABLE")
    zUnit = np.rint( (z-zBump+env_var("_Z_SNAP_BOOST")) / env_var("_BLOCK_SCALE") ) # Quantize to multiple of block unit length
    zBloc = max( (zUnit*env_var("_BLOCK_SCALE"))+zBump, zBump )
    return zBloc



########## LOGGER ##################################################################################

class LogPickler:
    """ Save Recordings as PKL files """

    def open_file( self ):
        """ Set the name of the current file """
        dateStr     = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        self.outNam = f"{self.prefix}_{dateStr}.pkl"
        if (self.outFil is not None) and (not self.outFil.closed):
            self.outFil.close()
        self.outFil = open( os.path.join( self.outDir, self.outNam ), 'wb' )


    def __init__( self, prefix = "Data-Log", outDir = None ):
        """ Set the file `prefix` and open a file """
        self.prefix = str( prefix )
        self.outDir = outDir if (outDir is not None) else '.'
        self.log    = deque()
        self.outFil = None
        self.open_file()


    def dump_to_file( self, openNext = False ):
        """ Write all data lines to a file """
        if len( self.log ):
            if (self.outFil is None) or self.outFil.closed:
                self.open_file()
            pickle.dump( list( self.log ), self.outFil )
            self.outFil.close()
            self.log = deque()
        if openNext:
            self.open_file()

    
    def append( self, datum = None, msg = None ):
        """ Add an item to the log """
        self.log.append( {
            't'    : now(),
            'msg'  : msg,
            'data' : datum,
        } )


def deep_copy_memory_list( mem : list[GraspObj] ):
    """ Make a deep copy of the memory list """
    rtnLst = list()
    for m in mem:
        rtnLst.append( m.copy() )
    return rtnLst



########## GEOMETRY FUNCTIONS ######################################################################

def closest_ray_points( A_org, A_dir, B_org, B_dir ):
    """ Return the closest point on ray A to ray B and on ray B to ray A """
    # https://palitri.com/vault/stuff/maths/Rays%20closest%20point.pdf
    c  = np.subtract( B_org, A_org )
    aa = np.dot( A_dir, A_dir )
    bb = np.dot( B_dir, B_dir )
    ab = np.dot( A_dir, B_dir )
    ac = np.dot( A_dir, c     )
    bc = np.dot( B_dir, c     ) 
    dv = (aa*bb-ab*ab)
    if dv < 0.0001:
        raise ValueError( f"BAD DIVISOR computed from: \n{A_org=}, {A_dir=}, \n{B_org=}, {B_dir=}" )
    fA = (-ab*bc + ac*bb) / dv
    fB = ( ab*ac - bc*aa) / dv
    pA = np.add( A_org, np.multiply( A_dir, fA ) )
    pB = np.add( B_org, np.multiply( B_dir, fB ) )
    return pA, pB