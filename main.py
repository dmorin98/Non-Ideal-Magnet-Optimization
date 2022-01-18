import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from scipy.optimize import minimize
import scipy.io

SIZE = 15
plt.rc('font', size=SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SIZE)    # legend fontsize
plt.rc('figure', titlesize=SIZE)  # fontsize of the figure title
mpl.rcParams['axes.linewidth'] = 2
plt.rcParams['font.family'] = 'Arial'

#Importing Field Profile
mat = scipy.io.loadmat('Width_Y_1D_Magnetic_field.mat')
Y = np.transpose(mat['y'])
B = (mat['B'])

#Magnet Dimensions
middle_w = 5
middle_h = 3

#Displacements
middle_vertical = 0
middle_horizontal = 0

#Parameters
x0 = [2400, 0, 0, 1, 0, 0, 1] #Remanence, a, b, c, a2, b2, c3

def B_z(z,y,z1,y1, co):
    #t = thickness of magnet
    #w = width of magnet
    #z1 = z displacement
    #y1 = y displacement
    #co = coefficient Rem, a b, c
    Rem = co[0]
    a = co[1]
    b = co[2]
    c = co[3]
    a2 = co[4]
    b2 = co[5]
    c2 = co[6]
    return Rem*((  y*(2*a*z+b)*np.log(((2*z-w)**2 + 4*y**2)/4) - y*(2*a*z+b)*np.log(((2*z+w)**2 + 4*y**2)/4)  )/2  +  ( z*(a*z+b) - a*y**2 + c )*np.arctan((2*z+w)/(2*y))  -  ( z*(a*z+b) -a*y**2 + c )*np.arctan((2*z-w)/(2*y))  +  a*w*y) \
        - Rem*((  (y+t)*(2*a2*z+b2)*np.log(((2*z-w)**2 + 4*(y+t)**2)/4) - (y+t)*(2*a2*z+b2)*np.log(((2*z+w)**2 + 4*(y+t)**2)/4)  )/2  +  ( z*(a2*z+b2) - a2*(y+t)**2 + c2 )*np.arctan((2*z+w)/(2*(y+t)))  -  ( z*(a2*z+b2) -a2*(y+t)**2 + c2 )*np.arctan((2*z-w)/(2*(y+t)))  +  a2*w*(y+t))

def B_y(z,y,z1,y1, co):
    Rem = co[0]
    a = co[1]
    b = co[2]
    c = co[3]
    a2 = co[4]
    b2 = co[5]
    c2 = co[6]
    return -Rem*((  (z*(a*z+b)-a*y**2+c) * ( np.log(((2*z+w)**2  +  4*y**2)/4) - np.log(((2*z-w)**2  +  4*y**2)/4) )  )/2  -  2*a*y*z*np.arctan((2*z+w)/(2*y))  -  b*y*np.arctan((2*z+w)/(2*y))  +  2*a*y*z*np.arctan((2*z-w)/(2*y))  +  b*y*np.arctan((2*z-w)/(2*y)) + a*w*z + b*w) \
        - Rem*(-(  (z*(a2*z+b2)-a2*(y+t)**2+c2) * ( np.log(((2*z+w)**2  +  4*(y+t)**2)/4) - np.log(((2*z-w)**2  +  4*(y+t)**2)/4) )  )/2  -  2*a2*(y+t)*z*np.arctan((2*z+w)/(2*(y+t)))  -  b2*(y+t)*np.arctan((2*z+w)/(2*(y+t)))  +  2*a2*(y+t)*z*np.arctan((2*z-w)/(2*(y+t)))  +  b2*(y+t)*np.arctan((2*z-w)/(2*(y+t))) + a2*w*z + b2*w)

z = np.linspace(-6, 6, len(Y))
y = np.linspace(min(Y), max(Y), len(Y))
Z, Y = np.meshgrid(z,y)

#Center Magnet
w=middle_w
t=middle_h
Bz = B_z(Z, Y, middle_horizontal, middle_vertical, x0)
By = B_y(Z, Y, middle_horizontal, middle_vertical, x0)
Bmag_m = np.sqrt(Bz**2 + By**2)

B_total = Bmag_m
goal_field = B
#Bottom left plot // Height of horizontal  line
y_measure = 200 #array index
print('Y Height:', y[y_measure])

plt.ion()
fig = plt.figure(figsize= (10,8))
plt.subplot(2,2,3)
fig.tight_layout(pad=4.0)
plt.plot(y, goal_field,'--r')
plt.title('Optimizated Magnetic Field')
plt.ylim(0.8*min(B), 1.2*max(B))
line1, = plt.plot(y, B_total[y_measure,:],color='k') # Returns a tuple of line objects, thus the comma

def plot(B_current):
    #Real Field
    line1.set_ydata(B_current[y_measure,:])
    plt.xlabel('z (cm)')
    plt.ylabel('B (gauss)')

    fig.canvas.draw()
    fig.canvas.flush_events()

#Starting Field Contour
plt.subplot(2,1,1)
plt.contour(Z, Y, B_total, 50)
plt.axhline(y=y[y_measure])
plt.xlabel('z (cm)')
plt.ylabel('y (cm)')

#Wanted Field
plt.subplot(2,2,4)
plt.plot(y, goal_field,color='k')
plt.title('Goal Experimental Magnetic Field')
plt.xlabel('z (cm)')
plt.ylabel('B (gauss)')
plt.ylim(0.8*min(B), 1.2*max(B))
plt.show()

#Simulation
residual = []
def objective(x):
    residuals = 0
    wanted_Field = goal_field
    Bz = B_z(Z, Y, middle_horizontal, middle_vertical, x)
    By = B_y(Z, Y, middle_horizontal, middle_vertical, x)
    B_current = np.sqrt(Bz**2 + By**2)
    B_total = Bmag_m

    for i in range (0, len(wanted_Field), 1):
        residuals += (np.abs(B_current[y_measure,i] - wanted_Field[i]))**2
    plot(B_current)
    print('Residuals:', residuals)
    residual.append(residuals)
    return residuals
    
sol = minimize(objective, x0, method='trust-constr')
print(sol.x)
plt.show()




