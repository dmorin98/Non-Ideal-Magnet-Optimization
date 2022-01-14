# Non-Ideal-Magnet-Optimization

## Description

The goal of this program is to provide a simple solution to describing non-ideal magnets (e.g. magnets with non-uniform remanence).
The program works by assuming the magnetic remenance varies as a function width along the magnet surface. Magnetic field equations for three infinitely long sheets of current are
used as an approximation of the magnetic fields of a permenant block magnet. The simulated spatial magnet remanence is modulated through a variation of parameter values 
to create the expressions which describe a non-ideal magnet. 

An experimental magnetic field plot along the width of a block magnet was measured, of which has a non-ideal remanence. This is found in the **/Width_Y_1D_Magnetic_field.mat/** file.
The optimization algorithm aims to minimize the sum of the squared residuals from the simulated and experimental field results. 

## Beginning of Optimization
![start_optimization](https://user-images.githubusercontent.com/52712406/149584199-65b7042a-96e7-47eb-ac5c-34eb1ef23eee.png)


## Finished Optimization
