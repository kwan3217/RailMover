# RailMover
The reference problem for this package is: 
Given a simulated roller coaster track, 
generate a set of simulate AHRS sensor data 
to test an AHRS filter.

## Parts
* `Rail` - Piecewise vector function with support
  for traversing along it by distance in addition
  to by parameter. Supports calculating
  derivative-related quantities such as curvature.
* `Track` - Piecewise scalar function for tagging
  sections along a track. For instance, 
  we can say that a particular part of the track
  is a booster, a brake, etc.
* `Mover` - an object which moves along a rail
  and generates simulated AHRS data.