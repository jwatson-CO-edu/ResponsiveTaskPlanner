########## INIT ####################################################################################

##### Imports #####

import math, time, os
now = time.time

from random import random

import numpy as np

from spatialmath.quaternion import UnitQuaternion
from spatialmath.base import r2q

from aspire.env_config import env_var



########## DEBUGGING ###############################################################################

def breakpoint( bpName ):
    """ Pause program at the terminal until the user presses [Enter] """
    return input( f"\n>>>> BREAKPOINT: {bpName} <<<<\n" )



########## CONTAINER OPERATIONS ####################################################################

def p_lst_has_nan( lst ):
    """ Does the list contain NaN? """
    for elem in lst:
        if math.isnan( elem ):
            return True
    return False


def p_list_duplicates( lst ):
    """ Return True if a value appears more than once """
    s = set( lst )
    return (len( lst ) > len( s ))


def sorted_obj_labels( obj ):
    """ Get the label dist keys in a PREDICTABLE ORDER """
    # WARNING: THIS FUNCTION BECOMES NECESSARY *AS SOON AS* GLOBAL LABLES ARE **NOT** FIXED!
    rtnLst = list( obj.labels.keys() )
    rtnLst.sort()
    return rtnLst


def match_name( shortName ):
    """ Search for the environment object name that matches the abbreviated query """
    for envName in env_var("_BLOCK_NAMES"):
        if shortName in envName:
            return envName
    return None


def extract_dct_values_in_order( dct, keyLst, insertZero = False ):
    """ Get the `dct` values in the order specified in `keyLst` """
    # NOTE: "TaskPlanner.py" version!
    rtnLst = []
    for k in keyLst:
        if k in dct:
            rtnLst.append( dct[k] )
        elif insertZero:
            rtnLst.append( 0.0 )
    return rtnLst


def extract_dct_values_in_order_conf( dct, keyLst, insertZero = False ):
    """ Get the `dct` values in the order specified in `keyLst` """
    # WARNING: "Confusion.py" VERSION! **BREAKS** THE PLANNER!
    rtnLst = []
    altNam = dict()
    for k in dct.keys():
        if match_name(k) in env_var("_BLOCK_NAMES"):
            altNam[ match_name(k) ] = k
    for k in keyLst:
        if k in dct:
            rtnLst.append( dct[k] )
        if k in altNam:
            rtnLst.append( dct[altNam[k]] )
        elif insertZero:
            rtnLst.append( 0.0 )
    return rtnLst


########## FILER OPERATIONS ########################################################################

def get_paths_in_dir_with_prefix( directory, prefix ):
    """ Get only paths in the `directory` that contain the `prefix` """
    fPaths = [os.path.join(directory, f) for f in os.listdir( directory ) if os.path.isfile( os.path.join(directory, f))]
    return [path for path in fPaths if (prefix in str(path))]




########## GEOMETRY ################################################################################

def pb_posn_ornt_to_homog( posn, ornt ):
    """ Express the PyBullet position and orientation as homogeneous coords """
    H = np.eye(4)
    Q = UnitQuaternion( ornt[-1], ornt[:3] )
    H[0:3,0:3] = Q.SO3().R
    H[0:3,3]   = np.array( posn )
    return H


def row_vec_to_pb_posn_ornt( V ):
    """ [Px,Py,Pz,Ow,Ox,Oy,Oz] --> [Px,Py,Pz],[Ox,Oy,Oz,Ow] """
    posn = np.array( V[0:3] )
    ornt = np.zeros( (4,) )
    ornt[:3] = V[4:7]
    ornt[-1] = V[3]
    return posn, ornt


def row_vec_to_homog( V ):
    """ Express [Px,Py,Pz,Ow,Ox,Oy,Oz] as homogeneous coordinates """
    posn, ornt = row_vec_to_pb_posn_ornt( V )
    return pb_posn_ornt_to_homog( posn, ornt )


def diff_norm( v1, v2 ):
    """ Return the norm of the difference between the two vectors """
    return np.linalg.norm( np.subtract( v1, v2 ) )


def closest_dist_Q_to_segment_AB( Q, A, B, includeEnds = True ):
    """ Return the closest distance of point Q to line segment AB """
    l = diff_norm( B, A )
    if l <= 0.0:
        return diff_norm( Q, A )
    D = np.subtract( B, A ) / l
    V = np.subtract( Q, A )
    t = V.dot( D )
    if (t > l) or (t < 0.0):
        if includeEnds:
            return min( diff_norm( Q, A ), diff_norm( Q, B ) )
        else:
            return float("NaN")
    P = np.add( A, D*t )
    return diff_norm( P, Q ) 



########## STATS & SAMPLING ########################################################################

def total_pop( odds ):
    """ Sum over all categories in the prior odds """
    total = 0
    for k in odds:
        total += odds[k]
    return total


def normalize_dist( odds_ ):
    """ Normalize the distribution so that the sum equals 1.0 """
    total  = total_pop( odds_ )
    rtnDst = dict()
    for k in odds_:
        rtnDst[k] = odds_[k] / total
    return rtnDst


def roll_outcome( odds ):
    """ Get a random outcome from the distribution """
    oddsNorm = normalize_dist( odds )
    distrib  = []
    outcome  = []
    total    = 0.0
    for o, p in oddsNorm.items():
        total += p
        distrib.append( total )
        outcome.append( o )
    roll = random()
    for i, p in enumerate( distrib ):
        if roll <= p:
            return outcome[i]
    return None


def get_confusion_matx( Nclass, confuseProb = 0.10 ):
    """ Get the confusion matrix from the label list """
    Pt = 1.0-confuseProb*(Nclass-1)
    Pf = confuseProb
    rtnMtx = np.eye( Nclass )
    for i in range( Nclass ):
        for j in range( Nclass ):
            if i == j:
                rtnMtx[i,j] = Pt
            else:
                rtnMtx[i,j] = Pf
    return rtnMtx


def multiclass_Bayesian_belief_update( cnfMtx, priorB, evidnc ):
    """ Update the prior belief using probabilistic evidence given the weight of that evidence """
    Nclass = cnfMtx.shape[0]
    priorB = np.array( priorB ).reshape( (Nclass,1,) )
    evidnc = np.array( evidnc ).reshape( (Nclass,1,) )
    P_e    = cnfMtx.dot( priorB ).reshape( (Nclass,) )
    P_hGe  = np.zeros( (Nclass,Nclass,) )
    for i in range( Nclass ):
        P_hGe[i,:] = (cnfMtx[i,:]*priorB[i,0]).reshape( (Nclass,) ) / P_e
    return P_hGe.dot( evidnc ).reshape( (Nclass,) )


def get_confused_class_reading( label, confProb, orderedLabels ):
    """ Return a discrete distribution with uniform confusion between classes other than `label` """
    rtnLabels = {}
    Nclass    = len( orderedLabels )
    for i in range( Nclass ):
        blkName_i = orderedLabels[i]
        if blkName_i == label:
            rtnLabels[ blkName_i ] = 1.0-confProb*(Nclass-1)
        else:
            rtnLabels[ blkName_i ] = confProb
    return rtnLabels


