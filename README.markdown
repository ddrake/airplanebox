Airplane Box
============

Suppose we have a box (inertial measurement unit - IMU) mounted on our airplane that is able to output a sequence of its Euler angles: roll, pitch and yaw with respect to the earth.  If the box is mounted in a perfect orientation (i.e. facing the front of the plane, and perfectly level) its output corresponds to the roll, pitch and yaw of the plane.  But if its mounting is not perfect, so that the box is rolled, pitched and/or yawed with respect to the plane, we will need to correct the data.

Two very important points needs to be made about the interpretation of roll pitch and yaw of the box relative to the airplane or the airplane relative to the earth:

1. R(roll, pitch, yaw) performs the roll first, then the pitch, and finally the yaw.  If the order of these operations is changed, a different rotation matrix would result.
+ Be careful when interpreting the rotations.  The yaw rotation of the airplane occurs about the z-axis of the world frame, not the body frame.

ANALYSIS
--------
The angles outputted by the box can be used to construct a rotation matrix T0 that takes any vector in the boxes' coordinate space into the global (earth) coordinate space. With the plane on the ground, level, facing North, suppose we mount the box with some roll, pitch and yaw relative to the aircraft.  If we turn on the box at this time, it should report precisely these values because the aircraft coordinate system is the same as the global one.

Now, once the aircraft is in flight, the box will output a sequence of triples of (roll, pitch and yaw).  We can construct a rotation matrix for each of these and call it Tk.  Each Tk maps vectors in box space to vectors in global space (Tk\*vbox -> vglobal).  

Our original T0 maps vectors in box space to vectors in plane space (T0\*vbox -> vplane).  Its inverse Inv(T0) maps vectors in plane space to vectors in box space (Inv(T0)\*vplane -> vbox). 

Therefore Tk\*Inv(T0) maps vectors in plane space to vectors in global space (what we want) since Tk\*(Inv(T0)\*vplane) = Tk\*vbox -> vglobal).  Once we've calculated this transformation matrix Tk\*Inv(T0), we can construct the Euler angles from it.

Things get interesting when the pitch is pi/2 or -pi/2.  When pitch is pi/2, the rotation matrix degenerates to:

    [[0,       sin(r-y),  cos(r-y)],
     [0,       cos(r-y),  -sin(r-y)],
     [-1,      0,         0]]

When pitch is -pi/2, the rotation matrix degenerates to:

    [[0,       -sin(r+y),  -cos(r+y)],
     [0,       cos(r+y),   -sin(r+y)],
     [1,       0,          0]]

Since for these cases, the rotation matrix is a function only of the sum or difference of the roll and yaw there is no way to solve for their individual values.  In general, the best we can do is pick an arbitrary value for one and calculate the other so that the sum or difference is correct.  However, if we are correcting a continuous sequence, we should be able to base our "arbitrary"  choice on the last known values of roll or yaw.  More importantly, it's easy to show that the spatial orientation is the same regardless of what arbitrary choice we make.

For example, suppose p = pi/2, sin(r-y) = .5 and cos(r-y) = .866
then we have r-y = pi/6 or r = y+pi/6.

(r,p,y) = pi/6, pi/2, 0 is precisely the same orientation as pi/3, pi/2, pi/6

Or if we suppose p = -pi/2, sin(r+y) = .5 and cos(r+y) = .866
then we have r+y = pi/6 or r = pi/6-y.

(r,p,y) = pi/6, pi/2, 0 is precisely the same orientation as 0, pi/2, pi/6

We can also consider the case when pi/2 < pitch < 3pi/2 separately.  In this case we can always construct an equivalent set of Euler angles (r', p', y') such that r' = r + pi, p' = pi - p, y' = y + pi.  We can prove their equivalence by substituting these values into the definition of the rotation matrix and observing that all nine terms of the matrix remain the same.

REFERENCES
----------
- <http://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles>
- <http://www.u.arizona.edu/~pen/ame553/Notes/Lesson%2009.pdf>
- <http://planning.cs.uiuc.edu/node102.html>

TESTS PROVIDED
--------------
Most functions in this script include doctests which will be run if the script is executed from the command line:

    > python airplanebox
or 

    > python airplanebox -v   (for verbose output)
    
The remaining functions are for ad hoc testing.  Not all the conversions are invertible, but some are.

If ea is the triple of Euler angles, T is the corresponding rotation martix,
ep is the tuple of Euler parameters, given some ea, we can compute:

ea -> T <-> ep -> ea' where ea' should "equal" the original set of Euler angles ea.  So two tests are:

1. converting a matrix T to Euler parameters, then recomputing the matrix from the parameters
+ starting with a set of Euler angles, computing T, ep, and then ea'. 

NOTE: ea' can in general be different from ea, but they should represent the same orientation.

