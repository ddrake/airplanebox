##    Dow Drake, 9/20/2011

import pylab
import sys
from numpy import *
from scipy.linalg import *
EPS = sys.float_info.epsilon

#
# GENERAL FUNCTIONS
#
def rotationMatrix(roll, pitch, yaw):
    """
    Return the rotation matrix when given the three Euler angles: roll, pitch and yaw.
    
    roll: cw rotation of the body about the x axis of the fixed frame (phi)
    pitch: cw rotation of the body about the y axis of the fixed frame (theta)
    yaw: cw rotation of the body about the z axis of the fixed frame (psi)
    This rotation matrix can be used to map any vector in the moving reference
    frame to the corresponding vector in the fixed frame by pre-multiplying it.
    
    By convention, the fixed z axis points down toward the center of the earth,
    the x axis points North and the y axis points East.
    A positive pitch is upward, a positive roll is to the right, a positive yaw is a right turn.

    >>> T = rotationMatrix(0,0,0)
    >>> import sys
    >>> abs(T-eye(3)) < EPS
    array([[ True,  True,  True],
           [ True,  True,  True],
           [ True,  True,  True]], dtype=bool)
    
    >>> T = rotationMatrix(pi/2,pi/2,pi/2)
    >>> abs(T-array([[0.,0,1],[0,1,0],[-1,0,0]])) < EPS
    array([[ True,  True,  True],
           [ True,  True,  True],
           [ True,  True,  True]], dtype=bool)

    >>> T = rotationMatrix(pi/4,0,0)
    >>> T
    array([[ 1.        ,  0.        ,  0.        ],
           [ 0.        ,  0.70710678, -0.70710678],
           [-0.        ,  0.70710678,  0.70710678]])
    >>> abs(T-array([[1.,0,0],[0,sqrt(2)/2,-sqrt(2)/2],[0,sqrt(2)/2,sqrt(2)/2]])) < EPS
    array([[ True,  True,  True],
           [ True,  True,  True],
           [ True,  True,  True]], dtype=bool)

    """
    T = empty((3,3))
    T[0,0]=cos(pitch)*cos(yaw)
    T[1,0]=cos(pitch)*sin(yaw)
    T[2,0]=-sin(pitch)
    T[0,1]=-cos(roll)*sin(yaw) + sin(roll)*sin(pitch)*cos(yaw)
    T[1,1]=cos(roll)*cos(yaw) + sin(roll)*sin(pitch)*sin(yaw)
    T[2,1]=sin(roll)*cos(pitch)
    T[0,2]=sin(roll)*sin(yaw) + cos(roll)*sin(pitch)*cos(yaw)
    T[1,2]=-sin(roll)*cos(yaw) + cos(roll)*sin(pitch)*sin(yaw)
    T[2,2]=cos(roll)*cos(pitch)
    return T

def eulerParameters(T):
    """
    Given a rotation matrix, compute the four Euler parameters.
    
    These parameters represent a 4-D vector that defines an axis such that if the
    xyz axes of the moving frame are rotated about it by an angle phi they will
    line up with the XYZ of the fixed frame and vice-versa.

    See: http://www.u.arizona.edu/~pen/ame553/Notes/Lesson%2009.pdf

    >>> es = eulerParameters(eye(3))
    >>> abs(array(es)-array([1.,0,0,0])) < EPS
    array([ True,  True,  True,  True], dtype=bool)

    >>> T = rotationMatrix(pi/4,0,0)
    >>> es = eulerParameters(T)
    >>> abs(array(es)-array((0.92387953251128674, 0.38268343236508973, 0.0, 0.0))) < EPS
    array([ True,  True,  True,  True], dtype=bool)
    """
    tra = trace(T)
    # take the positive square roots as a starting point
    e0 = sqrt(tra + 1.)/2.
    e1 = sqrt(1+2*T[0,0]-tra)/2.
    e2 = sqrt(1+2*T[1,1]-tra)/2.
    e3 = sqrt(1+2*T[2,2]-tra)/2.
    es = (e0,e1,e2,e3)
    eimax = es.index(max(es))

    # for best numerical stability, take the largest parameter as positive
    # and use it to re-compute the others with correct signs
    if eimax == 0:
        e1 = (T[2,1]-T[1,2])/4/e0
        e2 = (T[0,2]-T[2,0])/4/e0
        e3 = (T[1,0]-T[0,1])/4/e0
    elif eimax == 1:
        e0 = (T[2,1]-T[1,2])/4/e1
        e2 = (T[1,0]+T[0,1])/4/e1
        e3 = (T[0,2]+T[2,0])/4/e1
    elif eimax == 2:
        e0 = (T[0,2]-T[2,0])/4/e2
        e1 = (T[1,0]+T[0,1])/4/e2
        e3 = (T[2,1]+T[1,2])/4/e2            
    else:
        e0 = (T[1,0]-T[0,1])/4/e3
        e1 = (T[0,2]+T[2,0])/4/e3
        e2 = (T[2,1]+T[1,2])/4/e3            
    return (e0,e1,e2,e3)



def getEulerAngles(T):
    """
    Compute the roll, pitch and yaw from a rotation matrix.

    That function handles the singular cases directly and calls eulerParameters()
    for non-singular cases

    >>> angles = (pi/6,pi/2-2*EPS,-pi/4)
    >>> T = rotationMatrix(*angles)
    >>> newangles = getEulerAngles(T)
    >>> abs(angles[0]-angles[2]-newangles[0]) < EPS
    True
    >>> abs(angles[1]-newangles[1]) < 5*EPS
    True
    
    >>> angles = (pi/6,pi/2-5*EPS,-pi/4)
    >>> T = rotationMatrix(*angles)
    >>> es = eulerParameters(T)
    >>> newangles = eulerAngles(es)
    >>> err = abs(array(angles)-array(newangles))
    >>> err[0] < .03
    True
    >>> err[1] < 1.e-7
    True
    >>> err[2] < .06
    True
    """
    
    if abs(T[0,0]) < 4*EPS and abs(T[1,0]) < 4*EPS and abs(T[2,1]) < 4*EPS and abs(T[2,2]) < 4*EPS:
        if T[2,0] < 0. :
            p = pi/2
            y = 0
            r = arctan2(T[0,1],T[0,2])
        else:
            p = -pi/2
            y = 0
            r = arctan2(-T[0,1],-T[0,2])
        return (r,p,y)
    else:
        ep = eulerParameters(T)
        return eulerAngles(ep)
    

def eulerAngles(es):
    """
    Compute the roll, pitch and yaw from a tuple of the four Euler parameters.

    Note: this function is not intended to be used at the singular points
    where pitch = +/- pi/2.  In general, you should use getEulerAngles() instead.
    That function will handle the singular cases and call this function in the
    non-singular cases

    >>> angles = (pi/6,pi/2-2*EPS,-pi/4)
    >>> T = rotationMatrix(*angles)
    >>> es = eulerParameters(T)
    >>> newangles = eulerAngles(es)
    Traceback (most recent call last):
    ...
    AssertionError
    
    >>> angles = (pi/6,pi/2-5*EPS,-pi/4)
    >>> T = rotationMatrix(*angles)
    >>> es = eulerParameters(T)
    >>> newangles = eulerAngles(es)
    >>> err = abs(array(angles)-array(newangles))
    >>> err[0] < .03
    True
    >>> err[1] < 1.e-7
    True
    >>> err[2] < .06
    True
    """
    # http://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    # note arc
    e0, e1, e2, e3 = es
        
    
    dy_roll = 2*(e0*e1 + e2*e3)
    dx_roll = (1 - 2*(e1*e1 + e2*e2))
    dy_yaw = 2*(e0*e3 + e1*e2)
    dx_yaw = (1 - 2*(e2*e2 + e3*e3))
    # if both the dx and dy are tiny, the arctan is undefined.  Force it to zero to avoid numerical issues.
    # This can happen when the pitch is +/- pi/2 and the roll and yaw are both even or odd multiples of pi.
    # But forcing 
    assert (abs(dx_roll) > EPS or abs(dy_roll) > EPS) and (abs(dx_yaw) > EPS or abs(dy_yaw) > EPS)
    
    roll =  arctan2(dy_roll,dx_roll) if abs(dx_roll) > EPS or abs(dy_roll) > EPS else 0.
    pitch = arcsin(2*(e0*e2 - e3*e1)) 
    yaw =   arctan2(dy_yaw,dx_yaw) if abs(dx_yaw) > EPS or abs(dy_yaw) > EPS else 0.
    
    return (roll, pitch, yaw)




#
# FUNCTIONS SPECIFIC TO THE PROCESS OF CORRECTING THE REPORTED MEASUREMENTS 
#
def getCorrectionMatrix(roll, pitch, yaw):
    """
    Compute the correction matrix from the Euler angles of the box
    
    Given a set of Euler angles obtained by first mounting the box, then
    either temporarily positioning the airplane facing North on the ground and
    reading the output of the box -- or -- estimating the roll, pitch
    and yaw of the box relative to the stationary airplane by some other method.

    Return a correction matrix that can be used to calculate true roll, pitch
    and yaw from the measured values.
    
    >>> Tc = getCorrectionMatrix(.5, .5, .5)
    >>> abs(Tc - array([[ 0.77015115,  0.42073549, -0.47942554],
    ...                    [-0.21902415,  0.88034656,  0.42073549],
    ...                    [ 0.59907898, -0.21902415,  0.77015115]])) < 1.e-7
    array([[ True,  True,  True],
           [ True,  True,  True],
           [ True,  True,  True]], dtype=bool)

    
    """
    # note: we need to invert because we want a matrix that maps from the airplane
    # system to the box system
    return inv(rotationMatrix(roll, pitch, yaw))

def correctedEulerAngles(roll, pitch, yaw, Tc):
    """
    Compute the corrected Euler angles from measured values and the correction matrix
    
    Given a set of Euler angles (e.g. measurement data from the box)
    and a correction matrix Tc, which takes coordinates from the plane to the box,
    calculate a set of corrected Euler angles which represent the true roll, pitch
    and yaw of the airplane.

    >>> Tc = getCorrectionMatrix(.5, .5, .5)
    >>> ea = correctedEulerAngles(.9, -.2, .6, Tc)
    >>> abs(array(ea) - array((0.73182631479752158, -0.35738772904903454, -0.10266899093274005))) < EPS
    array([ True,  True,  True], dtype=bool)
    
    """
    # get the matrix which maps from the box to the world system (measured)
    Tk = rotationMatrix(roll, pitch, yaw)
    
    # T0 maps from the plane to the box system so multiplying takes us from plane to world.
    T1 = dot(Tk,Tc)
    
    es = eulerParameters(T1)
    result = eulerAngles(es)
    return result






#
# MAIN TEST PROCEDURE FOR VERIFYING THE CORRECTION PROCESS
#
    
def testCorrectionProcess(box_roll, box_pitch, box_yaw):
    """
    Given box mounting Euler angles, simulate raw and corrected output

    Sample Usage: testCorrectionProcess(.5, .5, .5)
    
    Given a set of Euler angles describing the mounting of the box
    Run through a simulation of sampled box data and display the corrected Euler angles.
    """
    print ('When the plane is level, facing North, the box, as mounted,'
        ' reports:\nroll=%f, pitch=%f and yaw=%f\n') %(box_roll, box_pitch, box_yaw)
    T = getCorrectionMatrix(box_roll, box_pitch, box_yaw)
    testnum = 1
    x = linspace(-.5,.5,3)
    print 'correction matrix to pre-multiply the rotation matrix:'
    print '(the rotation matrix is constructed from sampled box measurements during flight)'
    print T
    print
    for r in x:
        for p in x:
            for y in x:
                roll, pitch,yaw = correctedEulerAngles(r,p,y,T)
                print '%d. box reports: \troll=%f\tpitch=%f\tyaw=%f' %(testnum, r, p, y)
                print 'corrected values: \troll=%f\tpitch=%f\tyaw=%f' %(roll, pitch,yaw)
                print
                testnum += 1
    


#
# TEST THE RANGE OF THE ALGORITHM
#
def testGetEulerAngles(xr_min=-pi, xr_max=pi, nxr=20,
                       xp_min=0, xp_max=pi/2, nxp=20,
                       xy_min=-pi, xy_max=pi, nxy=20,
                       verbose=False, eps=1.e-8):
    """
    Run tests over a range of Euler angles to assess the accuracy of getEulerAngles()

    Optionally specify the min, max and number of points for roll, pitch and yaw
    Optionally specfiy verbose output and/or the tolerance eps.
    Take a set of Euler angles and compute their associated rotation matrix, then use that
    rotation matrix to re-compute the original Euler angles and see if they match.
    Not matching does not necessarily mean that the algorithm has failed, since many different sets
    of Euler angles can be used to represent the same orientation.
    """
    failedCount = 0
    allCount = 0
    verbose = False
    points = 150
    xr = linspace(xr_min, xr_max, nxr)
    xp = linspace(xp_min, xp_max, nxp)
    xy = linspace(xy_min, xy_max, nxy)
    maxerr = 0
    for r in xr:
        if not verbose: print '.',
        for p in xp:
            for y in xy:
                failed, (re,pe,ye), (cr,cp,cy) = testGetEulerAngles_helper(r,p,y,eps,verbose)
                allCount +=1
                if failed: failedCount += 1
                tmax = max(abs(re),abs(pe),abs(ye))
                maxerr = tmax if tmax > maxerr else maxerr

    print                
    print "Total Differences greater than eps: %d out of %d trials" %(failedCount, allCount)
    print "Largest Difference: %e" %(maxerr)
    

def testGetEulerAngles_helper(roll, pitch, yaw, eps=EPS, verbose=False):
    """
    Create a rotation matrix from some Euler angles and use it to re-compute the original Euler angles

    This function can be called directly for individual tests.
    It's also called by testGetEulerAngles() to look for failures in a range of combinations
    """
    T = rotationMatrix(roll, pitch, yaw)
    angles = getEulerAngles(T)
    angles = tuple([standardAngle(x) for x in angles])
    angleCompare = zip((roll,pitch,yaw),angles)
    angleErrs = tuple([standardAngle(x-y) for x,y in angleCompare])
    r, p, y = angles
    re,pe,ye = angleErrs
    failed = False
    
    if abs(re) > eps or abs(pe) > eps or abs(ye) > eps:
        failed = True
        if verbose: 
            print
            print 'RESULT DIFFERS FROM INPUT'
            print 'GIVEN      roll=%f, pitch=%f, yaw=%f' %(roll, pitch, yaw)
            print 'CALCULATED roll=%f, pitch=%f, yaw=%f' %angles
            print 'roll-r = %e, pitch-p = %e, yaw-y = %e' %angleErrs
    return (failed,angleErrs,angles)

def standardAngle(angle):
    """
    Given an angle in radians, convert it to an angle in the range -pi < result <= pi

    Note: if the original angle is many multiples of 2pi outside the range, the error can be
    significant.

    >>> abs(standardAngle(pi)-pi) < EPS
    True
    >>> abs(standardAngle(-pi)+pi) < EPS
    True
    >>> abs(standardAngle(-pi+EPS)+pi-EPS) < EPS
    False
    >>> abs(standardAngle(-pi+EPS)-(-pi+EPS)) < EPS
    True
    >>> abs(standardAngle(pi+EPS)-(pi+EPS)) < EPS
    True
    >>> abs(standardAngle(pi+2*EPS)-(pi+2*EPS)) < EPS
    False
    >>> abs(standardAngle(pi+2*EPS)-(-pi+2*EPS)) < EPS
    True
    >>> abs(standardAngle(EPS)-(EPS)) < EPS
    True
    >>> abs(standardAngle(-EPS)-(-EPS)) < EPS
    True
    >>> abs(standardAngle(13*pi/6)-(pi/6)) < 2*EPS
    True
    """
    if abs(angle-pi) < EPS:
        return pi
    if abs(angle+pi) < EPS:
        return -pi
    
    n = round(angle/2./pi)
    return angle - 2*n*pi





#
# PRELIMINARY TESTS
#
def testGetSameMatrix(roll, pitch, yaw, verbose=False):
    """
    Test the eulerParameters() function by using the parameters to re-generate the matrix
    
    Create a rotation matrix from some Euler angles, then
    get the Euler parameters for that rotation matrix, then
    use these parameters to re-compute the rotation matrix
    
    >>> testGetSameMatrix(.5,pi/2,.5)
    array([[ True,  True,  True],
           [ True,  True,  True],
           [ True,  True,  True]], dtype=bool)
    """
    T1 = rotationMatrix(roll, pitch, yaw)
    es = eulerParameters(T1)
    T2 = rotationMatrix_FromEulerParameters(es)
    if verbose:
        print T1
        print es
        print T2
    return abs(T1-T2) < EPS

def testGetSameMatrix_homogeneous(roll, pitch, yaw, verbose=False):
    """
    Test the eulerParameters() function by using the parameters to re-generate the matrix
    
    Create a rotation matrix from some Euler angles, then
    get the Euler parameters for that rotation matrix, then
    use these parameters to re-compute the rotation matrix
    
    >>> testGetSameMatrix_homogeneous(.5,pi/2,.5)
    array([[ True,  True,  True],
           [ True,  True,  True],
           [ True,  True,  True]], dtype=bool)
    """
    T1 = rotationMatrix(roll, pitch, yaw)
    es = eulerParameters(T1)
    T2 = rotationMatrix_FromEulerParameters_homogeneous(es)
    if verbose:
        print T1
        print es
        print T2
    return abs(T1-T2) < EPS

def rotationMatrix_FromEulerParameters(es):
    """
    Construct a non-homogeneous rotation matrix from the Euler parameters
    
    Given a set of Euler parameters, compute the rotation matrix.
    This can be helpful as an intermediate step in testing that the
    Euler parameters were calculated correctly from the rotation matrix

    >>> T=rotationMatrix(.5,.5,.5)
    >>> ep = eulerParameters(T)
    >>> T1=rotationMatrix_FromEulerParameters(ep)
    >>> abs(T-T1) < EPS
    array([[ True,  True,  True],
           [ True,  True,  True],
           [ True,  True,  True]], dtype=bool)
    """
    e0,e1,e2,e3 = es
    T = empty((3,3))
    T[0,0] = 1 - 2*(e2*e2 + e3*e3)
    T[1,0] = 2*(e1*e2 + e0*e3)
    T[2,0] = 2*(e1*e3 - e0*e2)
    
    T[0,1] = 2*(e1*e2 - e0*e3)
    T[1,1] = 1 - 2*(e1*e1 + e3*e3)
    T[2,1] = 2*(e0*e1 + e2*e3)
    
    T[0,2] = 2*(e0*e2 + e1*e3)
    T[1,2] = 2*(e2*e3 - e0*e1)
    T[2,2] = 1 - 2*(e1*e1 + e2*e2)    
    return T

def rotationMatrix_FromEulerParameters_homogeneous(es):
    """
    Construct a homogeneous rotation matrix from the Euler parameters

    Given a set of Euler parameters, compute the rotation matrix.
    This can be helpful as an intermediate step in testing that the
    Euler parameters were calculated correctly from the rotation matrix
    
    >>> T=rotationMatrix(.5,.5,.5)
    >>> ep = eulerParameters(T)
    >>> T1=rotationMatrix_FromEulerParameters_homogeneous(ep)
    >>> abs(T-T1) < EPS
    array([[ True,  True,  True],
           [ True,  True,  True],
           [ True,  True,  True]], dtype=bool)
    """
    e0,e1,e2,e3 = es
    T = empty((3,3))
    T[0,0] = e0*e0 + e1*e1 - e2*e2 - e3*e3
    T[1,0] = 2*(e1*e2 + e0*e3)
    T[2,0] = 2*(e1*e3 - e0*e2)
    
    T[0,1] = 2*(e1*e2 - e0*e3)
    T[1,1] = e0*e0 - e1*e1 + e2*e2 - e3*e3
    T[2,1] = 2*(e0*e1 + e2*e3)
    
    T[0,2] = 2*(e0*e2 + e1*e3)
    T[1,2] = 2*(e2*e3 - e0*e1)
    T[2,2] = e0*e0 - e1*e1 - e2*e2 + e3*e3
    return T

    
def testRotationMatrix():
    """
    Basic test of rotationMatrix()
    
    Try to verify that the rotationMatrix function does what it's supposed to
    by calculating it for various roll, pitch, yaw values, then post-multiplying it by
    some basis vectors.
    """
    print 'TEST0: Try with all zero angles'
    print 'Expect the identity matrix'
    T =  rotationMatrix(0.,0.,0.)
    print T
    print
    print

    print 'TEST1: Try a small positive yaw.'
    T =  rotationMatrix(0,0,.1)
    print 'the matrix:'
    print T
    vx = array(((1,0,0,))).transpose()
    print 'unit vector in x direction in airplane system'
    print vx
    print 'Expect large positive x, small positive y, zero z'
    print 'rotated vector is:'
    print dot(T,vx)
    print
    vy = array(((0,1,0,))).transpose()
    print 'unit vector in y direction in airplane system'
    print vy
    print 'Expect small negative x, large positive y, zero z'
    print 'rotated vector is:'
    print dot(T,vy)
    print
    vz = array(((0,0,1,))).transpose()
    print 'unit vector in z direction in airplane system'
    print vz
    print 'Expect zero x, zero y, unit z'
    print 'rotated vector is:'
    print dot(T,vz)
    print
    print
    
    print 'TEST2: Try a small positive pitch.'
    T =  rotationMatrix(0,.1,0)
    print 'the matrix:'
    print T
    vx = array(((1,0,0,))).transpose()
    print 'unit vector in x direction in airplane system'
    print vx
    print 'Expect large positive x, zero y, small negative z (-z is up!)'
    print 'rotated vector is:'
    print dot(T,vx)
    print
    vy = array(((0,1,0,))).transpose()
    print 'unit vector in y direction in airplane system'
    print vy
    print 'Expect zero x, unit y, zero z'
    print 'rotated vector is:'
    print dot(T,vy)
    print
    vz = array(((0,0,1,))).transpose()
    print 'unit vector in z direction in airplane system'
    print vz
    print 'Expected small negative x, zero y, large positive z'
    print 'I was wrong about the x -- it''s small positive because'
    print 'the z axis points down!'
    print 'rotated vector is:'
    print dot(T,vz)
    print
    print
    
    print 'TEST3: Try a small positive roll.'
    T =  rotationMatrix(.1,0,0)
    print 'the matrix:'
    print T
    vx = array(((1,0,0,))).transpose()
    print 'unit vector in x direction in airplane system'
    print vx
    print 'Expect unit x, zero y, zero z'
    print 'rotated vector is:'
    print dot(T,vx)
    print
    vy = array(((0,1,0,))).transpose()
    print 'unit vector in y direction in airplane system'
    print vy
    print 'Expect zero x, large positive y, small positive z'
    print 'rotated vector is:'
    print dot(T,vy)
    print
    vz = array(((0,0,1,))).transpose()
    print 'unit vector in z direction in airplane system'
    print vz
    print 'Expect zero x, small negative y, large positive z'
    print 'rotated vector is:'
    print dot(T,vz)
    print
    print
    
    print 'TEST4: combine small positive roll, pitch and yaw.'
    T =  rotationMatrix(.1,.1,.1)
    print 'the matrix:'
    print T
    vx = array(((1,0,0,))).transpose()
    print 'unit vector in x direction in airplane system'
    print vx
    print 'Expect large positive x, small positive y, small negative z'
    print 'rotated vector is:'
    print dot(T,vx)
    print
    vy = array(((0,1,0,))).transpose()
    print 'unit vector in y direction in airplane system'
    print vy
    print 'Expect small negative x, large positive y, small positive z'
    print 'rotated vector is:'
    print dot(T,vy)
    print
    vz = array(((0,0,1,))).transpose()
    print 'unit vector in z direction in airplane system'
    print vz
    print 'Expect small positive x, small negative y, large positive z'
    print 'rotated vector is:'
    print dot(T,vz)
    print
    
    print 'TEST5: combine 30 degree positive roll, 89 degree pitch and 0 yaw.'
    T =  rotationMatrix(pi/6.,89.*pi/180,0.)
    print 'the matrix:'
    print T
    vx = array(((1,0,0,))).transpose()
    print 'unit vector in x direction in airplane system'
    print vx
    print 'Expect zero x, small positive y, large negative z'
    print 'rotated vector is:'
    print dot(T,vx)
    print
    vy = array(((0,1,0,))).transpose()
    print 'unit vector in y direction in airplane system'
    print vy
    print 'Expect zero x, large positive y, small negative z'
    print 'rotated vector is:'
    print dot(T,vy)
    print
    vz = array(((0,0,1,))).transpose()
    print 'unit vector in z direction in airplane system'
    print vz
    print 'Expect large positive x, zero y, zero z'
    print 'rotated vector is:'
    print dot(T,vz)
    print
    

if __name__ == "__main__":
    import doctest
    doctest.testmod()
