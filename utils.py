import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt

#Defining the lattice params in Hexagonal notation
a_H = 4.9525
b_H = 4.9525
c_H = 14.00093
V_H = np.sin(60*np.pi/180)*a_H*b_H*c_H

a_M = 7.266
b_M = 5.0024
c_M = 5.5479
beta_M = 96.760
V_M = a_M*b_M*c_M*np.sin(beta_M*np.pi/180)


def apply_rotation(initial_matrix, rotx = 0, roty = 0, rotz = 0):
    rotation_matrix_x = np.array([
        [1, 0, 0],
        [0, np.cos(rotx * np.pi / 180), -np.sin(rotx * np.pi / 180)],
        [0, np.sin(rotx * np.pi / 180),  np.cos(rotx * np.pi / 180)]
    ])

    rotation_matrix_y = np.array([
        [np.cos(roty * np.pi / 180), 0, np.sin(roty * np.pi / 180)],
        [0, 1, 0],
        [-np.sin(roty * np.pi / 180), 0, np.cos(roty * np.pi / 180)]
    ])

    rotation_matrix_z = np.array([
        [np.cos(rotz * np.pi / 180), -np.sin(rotz * np.pi / 180), 0],
        [np.sin(rotz * np.pi / 180),  np.cos(rotz * np.pi / 180), 0],
        [0, 0, 1]
    ])

    #total_rotation = np.dot(rotation_matrix_z, np.dot(rotation_matrix_y, rotation_matrix_x))
    #rotated_matrix = np.dot(total_rotation,initial_matrix.T).T

    total_rotation = np.dot(rotation_matrix_x, np.dot(rotation_matrix_y, rotation_matrix_z))
    rotated_matrix = np.dot(initial_matrix, total_rotation)


    return rotated_matrix

"""
def forbidden_reflections(phase, hkl):
    h = hkl[0]
    k = hkl[1]
    l = hkl[2]
    
    if (h == k) and (k == l) and (l== 0):
        return True #Ruling out the [0,0,0]
    
    if phase == "Hexagonal":
        "Rules for space group 167"
        if (h == k) and (l%3 != 0):
            return True
        if (h == -k) and ((h+l)%3 != 0) and (l%2 != 0):
            return True
        if ((-h + k + l)%3 != 0):
            return True
        if ((h -k + l)%3 != 0):
            return True
    else:
        pass
        
    if phase == "Monoclinic":    
        "Rules for space group 15, representation 13"
        if (h == k) and (k == 0) and (l%2 != 0):
            return True
        if (h == l) and (l == 0) and (k%2 != 0):
            return True
        if (k == l) and (l == 0) and (h%2 != 0):
            return True
        if (h == 0) and ((k+l)%2 != 0):
            return True
        if (k == 0) and (h%2 != 0) and (l%2 != 0):
            return True
        if (l == 0) and ((h+k)%2 != 0):
            return True
        if ((h+k+l)%2 != 0):
            return True
    else:
        pass
        
    return False

def create_possible_reflections(phase, smallest_number, largest_number):

    rango_hkl = range(smallest_number, largest_number + 1)
    combinaciones_hkl = []
    for h in rango_hkl:
        for k in rango_hkl:
            for l in rango_hkl:
                hkl = [h,k,l]
                if forbidden_reflections(phase, hkl) == False:
                    combinaciones_hkl.append(np.array([hkl[0], hkl[1], hkl[2]]))
                else:
                    pass
    combinaciones_hkl = np.array(combinaciones_hkl)
    return combinaciones_hkl

"""    

def allowed_reflections(phase, hkl):
    h = hkl[0]
    k = hkl[1]
    l = hkl[2]

    dummy = []
    
    if (h == k == l == 0):
        return False #Ruling out the [0,0,0]
    
    if phase == "Hexagonal":
        "Rules for space group 167"

        #General Conditions
        if ((-h + k + l)%3 == 0) and (h or k != 0) and (l%2 == 0):
            dummy.append(1)
        if ((h - k + l)%3 == 0) and (h or k != 0) and (l%2 == 0):
            dummy.append(1)
        if (h == k != l) and (l%3 == 0) and (h == k != 0):
            dummy.append(1)
        if (h == -k) and ((h+l)%3 == 0) and (h or k != 0):
            dummy.append(1)
        if (h == -k) and ((k+l)%3 == 0) and (h or k != 0):
            dummy.append(1)

        # Special condition
        if (h == k == 0) and (l%2 == 0) and (l%3 == 0):
            dummy.append(1)
    
    if 1 in dummy:
        return True

def create_possible_reflections(phase, smallest_number, largest_number):

    rango_hkl = range(smallest_number, largest_number + 1)
    combinaciones_hkl = []
    for h in rango_hkl:
        for k in rango_hkl:
            for l in rango_hkl:
                hkl = [h,k,l]
                if allowed_reflections(phase, hkl) == True:
                    combinaciones_hkl.append(np.array([hkl[0], hkl[1], hkl[2]]))
                else:
                    pass
    combinaciones_hkl = np.array(combinaciones_hkl)
    return combinaciones_hkl

def cal_reciprocal_vector(vector1, vector2, vector3):

    volume = np.dot(vector1, np.cross(vector2, vector3))
    rec_vec1 = (2*np.pi/volume)*np.cross(vector2, vector3)
    rec_vec2 = (2*np.pi/volume)*np.cross(vector3, vector1)
    rec_vec3 = (2*np.pi/volume)*np.cross(vector1, vector2)

    return rec_vec1, rec_vec2, rec_vec3

def calculate_Q_hkl(hkl, a_rec, b_rec, c_rec):

    Q_hkl = hkl[0]*a_rec + hkl[1]*b_rec + hkl[2]*c_rec

    return Q_hkl

def check_Bragg_condition(Q_hkl, wavelength, E_bandwidth):

    ewald_sphere = Ewald_Sphere(wavelength, E_bandwidth)

    ki = np.array([2*np.pi/wavelength,0,0])

    kf = Q_hkl + ki
    kf_hkl_module = np.linalg.norm(kf)

    if (kf_hkl_module >= ewald_sphere.Get_Inner_Radius()) and (kf_hkl_module <= ewald_sphere.Get_Outer_Radius()):
        Bragg_Condition = True
    else:
        Bragg_Condition = False
            
    return Bragg_Condition

def diffraction_direction(Q_hkl, wavelength, sample_detector_distance, tilting_angle):

    ki = np.array([2*np.pi/wavelength, 0, 0])
    kf =  Q_hkl + ki

    dy = kf[1]*(wavelength/(2*np.pi))*sample_detector_distance
    dz = kf[2]*(wavelength/(2*np.pi))*sample_detector_distance


    if kf[0] > 0:
        dx = 1
    else:
        dx = 0

    
    return (dx, dy, dz)

def diffraction_in_detector(dx, dy, dz, max_detectable_z, max_detectable_y, margin = 0):
    safety = 1 - margin/100 #% of margin from the detector edges
    if dx > 0:
        if (dz <= max_detectable_z*(safety) and dz >= max_detectable_z*(1-safety)) and (dy <= max_detectable_y*(safety) and dy >= -max_detectable_y*(safety)):
            return True
        else:
            False

    else:
        return False
    

def calculate_two_theta_angle_laue(Q, wavelength):
    Q_norm = np.linalg.norm([Q[0], Q[1], Q[2]])
    two_theta = 2*np.arcsin(Q_norm*wavelength/(4*np.pi))*180/np.pi

    return two_theta

def calculate_two_theta_angle(phase, hkl, wavelength):

    if phase == "Monoclinic":
        lattice_structure = Monoclinic_Lattice()
    
    elif phase == "Hexagonal":
        lattice_structure = Hexagonal_Lattice()

    a, b, c = lattice_structure.crystal_orientation

    a_rec, b_rec, c_rec = cal_reciprocal_vector(a, b, c)

    Q = calculate_Q_hkl(hkl, a_rec, b_rec, c_rec)

    Q_norm = np.linalg.norm([Q[0], Q[1], Q[2]])
    two_theta = 2*np.arcsin(Q_norm*wavelength/(4*np.pi))*180/np.pi

    return two_theta

def atomic_structure_factor(atomic_structure_factor_list, wavelength, two_theta):
    sin_theta_over_lambda = np.sin((np.pi/180)*two_theta/2)/wavelength
    dummy = np.argmin(np.abs(atomic_structure_factor_list[:,0] - sin_theta_over_lambda))
    atomic_structure_factor = atomic_structure_factor_list[dummy][1]
    return atomic_structure_factor

def structure_factor_given_atom(atomic_factor, hkl, atomic_positions):
    dummy = []
    for atomic_position in atomic_positions:
        bla = atomic_factor*np.exp(2j*np.pi*np.dot(hkl, atomic_position))
        dummy.append(bla*np.conj(bla))
    structure_factor_atom = sum(dummy)

    return structure_factor_atom


def calculate_angle_between_two_reflections(phase, hkl_1, hkl_2):

    hkl_1 = np.array(hkl_1)
    hkl_2 = np.array(hkl_2)

    if phase == "Monoclinic":
        lattice_structure = Monoclinic_Lattice()
    
    elif phase == "Hexagonal":
        lattice_structure = Hexagonal_Lattice()


    a,b,c = lattice_structure.Get_Crystal_Orientation()
    a_rec, b_rec, c_rec = cal_reciprocal_vector(a, b, c)

    Q_hkl_1 = calculate_Q_hkl(hkl_1, a_rec, b_rec, c_rec)

    Q_hkl_2 = calculate_Q_hkl(hkl_2, a_rec, b_rec, c_rec)
    

    angle = np.arccos(np.dot(Q_hkl_1, Q_hkl_2)/(np.linalg.norm(Q_hkl_1)*np.linalg.norm(Q_hkl_2)))*180/np.pi

    return angle


class Lattice_Structure:
    def __init__(self, initial_crystal_orientation, Vanadium_fractional_position, Oxygen_fractional_position):
        self.initial_crystal_orientation = initial_crystal_orientation
        self.Vanadium_fractional_position = Vanadium_fractional_position
        self.Oxygen_fractional_position = Oxygen_fractional_position
        self.crystal_orientation = initial_crystal_orientation
    
    def Apply_Rotation(self, rotx = 0, roty = 0, rotz = 0):
        self.crystal_orientation = apply_rotation(self.initial_crystal_orientation, rotx, roty, rotz)
        #self.crystal_orientation = apply_rotation(self.crystal_orientation, rotx, roty, rotz)
    
class Hexagonal_Lattice(Lattice_Structure):
    def __init__(self):
        initial_crystal_orientation = np.array([
        [a_H     , 0               , 0  ],
        [-0.5*b_H, np.sqrt(3)*b_H/2, 0  ],
        [0       , 0               , c_H]
        ])

        Vanadium_fractional_position = np.array([0.00000,    0.00000,    0.34670])
        Oxygen_fractional_position   = np.array([0.31480,    0.00000,    0.25000])

        super().__init__(initial_crystal_orientation, Vanadium_fractional_position, Oxygen_fractional_position)

class Monoclinic_Lattice(Lattice_Structure):
    def __init__(self):
        initial_crystal_orientation = np.array([
        [b_M/2 - np.sqrt((c_M**2 + a_M*c_M*np.cos(beta_M*np.pi/180) - (b_M**2)/4)/3) ,b_M/np.sqrt(12) - np.sqrt(c_M**2 + a_M*c_M*np.cos(beta_M*np.pi/180) - (b_M**2)/4)    ,  np.sqrt((14*c_M**2 - 4*a_M**2 + a_M*c_M*np.cos(beta_M*np.pi/180))/9)],
        [b_M                                                                         ,0                                                                                    ,                                                                     0],
        [b_M/4 - np.sqrt((c_M**2 + a_M*c_M*np.cos(beta_M*np.pi/180) - (b_M**2)/4)/12),b_M/np.sqrt(48) - 0.5*np.sqrt(c_M**2 + a_M*c_M*np.cos(beta_M*np.pi/180) - (b_M**2)/4), -np.sqrt((14*c_M**2 - 4*a_M**2 + a_M*c_M*np.cos(beta_M*np.pi/180))/9)]
        ])

        Vanadium_fractional_position = np.array([0.34460,    0.00500,    0.29850])
        Oxygen_fractional_position   = np.array([[0.41100,    0.84700,    0.65000], [0.25000,    0.18000,    0.00000]])

        super().__init__(initial_crystal_orientation, Vanadium_fractional_position, Oxygen_fractional_position)

class Detector:
    def __init__(self, detector_type, sample_detector_distance = None, tilting_angle = None):

        self.detector_type = detector_type #I just want to make it accessible 

        if self.detector_type == "Rayonix":
            self.height = 170 #mm
            self.width = 170 #mm


        elif self.detector_type == "Pilatus":
            self.height = 142 #mm
            self.width = 253 #mm
        
        elif self.detector_type == "Rigaku":
            self.height = 80
            self.width  = 80
        
        else:
            raise ValueError("Invalid Detector. Choose between Rayonix and Pilatus")
        
        if sample_detector_distance is not None:
            self.sample_detector_distance = sample_detector_distance
        
        if tilting_angle is not None:
            self.tilting_angle = tilting_angle
        
    def Max_Detectable_Z(self):
        max_theta_z = np.arctan(np.cos(self.tilting_angle*np.pi/180)/((self.sample_detector_distance/self.height) - np.sin(self.tilting_angle*np.pi/180)))
        dz = self.sample_detector_distance*np.tan(max_theta_z)/(np.cos(self.tilting_angle*np.pi/180) + np.tan(max_theta_z)*np.sin(self.tilting_angle*np.pi/180))
        return dz
    
    def Max_Detectable_Y(self):
        max_theta_y = np.arctan(self.width/(2*self.sample_detector_distance))
        dy = self.sample_detector_distance*np.tan(max_theta_y)
        return dy
        
class Ewald_Sphere:
    def __init__(self, wavelength, E_bandwidth):
        self.wavelength = wavelength
        self.E_bandwidth = E_bandwidth
        self.radius = 2*np.pi/wavelength
        self.radius_inner = self.radius*(1 - (E_bandwidth/2)/100)
        self.radius_outer = self.radius*(1 + (E_bandwidth/2)/100)

    def Get_Radius(self):
        return self.radius

    def Get_Inner_Radius(self):
        return self.radius_inner
        
    def Get_Outer_Radius(self):
        return self.radius_outer

    def Generate_Ewald_Sphere_Data(self):
        theta = np.linspace(0, 2 * np.pi, 100)
        phi = np.linspace(0, np.pi, 100)
        theta, phi = np.meshgrid(theta, phi)

        x = self.radius*np.sin(phi)*np.cos(theta)
        y = self.radius*np.sin(phi)*np.sin(theta)
        z = self.radius*np.cos(phi)

        return x, y, z

    def Add_To_Existing_Plot(self, existing_fig):

        x, y, z = self.Generate_Ewald_Sphere_Data()

        sphere = go.Surface(
            x = x,
            y = y, 
            z = z,
            opacity = 0.2,  
            showscale = False,
            colorscale = "Blues"  
        )
        existing_fig.add_trace(sphere)

        return existing_fig
    
class Atom:
    def __init__(self, symbol, atomic_structure_factor):
        self.Symbol = symbol
        self.Atomic_Structure_Factor = atomic_structure_factor

class Vanadium(Atom):
    def __init__(self):
        symbol = "V"
        atomic_structure_factor = np.array([
            (0.00, 23.00),
            (0.05, 22.20),
            (0.10, 20.47),
            (0.15, 18.67),
            (0.20, 17.02),
            (0.25, 15.48),
            (0.30, 14.03),
            (0.35, 12.69),
            (0.40, 11.51),
            (0.45, 10.49),
            (0.50,  9.63),
            (0.60,  8.34),
            (0.70,  7.48),
            (0.80,  6.87),
            (0.90,  6.38),
            (1.00,  5.94),
            (1.10,  5.52),
            (1.20,  5.11),
            (1.30,  4.70),
            (1.40,  4.30),
            (1.50,  3.92),
            (1.60,  3.57),
            (1.70,  3.24),
            (1.80,  2.96),
            (1.90,  2.70),
            (2.00,  2.48)
        ])
        np.array(atomic_structure_factor)
        super().__init__(symbol, atomic_structure_factor)

class Oxygen(Atom):
    def __init__(self):
        symbol = "O"
        atomic_structure_factor = np.array([
            (0.00, 8.00),
            (0.05, 7.79),
            (0.15, 6.48),
            (0.20, 5.60),
            (0.25, 4.76),
            (0.30, 4.08),
            (0.35, 3.46),
            (0.40, 3.00),
            (0.45, 2.61),
            (0.50, 2.31),
            (0.55, 2.10),
            (0.60, 1.94)
        ])
        super().__init__(symbol, atomic_structure_factor)

def colorize(vector, vmin=None, vmax=None, ax=None, cmap=plt.cm.jet):
    """
    Function for redoing the color of the plots. The main idea is to declare a list, 
    transform it into a vector, and then use its values to create the cmap to color the plots.

    Parameters:
    ----------
    vector: list or tuple
        List or tuple with the indices of what is to be plotted.
    
    Optional:
    ---------
    vmin: Int or Float
        Default = None
        It sets the minimum index value for the cmap.
    vmax: Int or Float
        Default = None
        It sets the maximum index value for the cmap.
    ax: Int or Float
        Default = None
        It helps identifying the plot to be colorized.
    cmap: matplotlib.pyplot color set.
        Default = plt.cm.jet
        It selects the specific set of colors to be used.

    Returns:
    ---------
    Nothing
    """

    vector = np.asarray(vector)
    # get plot
    if ax is None:
        ax = plt.gca()
    # normalize vector
    if vmin is None:
        vmin = vector.min()
    if vmax is None:
        vmax = vector.max()
    vector = (vector - vmin) / (vmax - vmin)

    for i, value in enumerate(vector):
        ax.collections[i].set_color(cmap(value))



#hkls = [[1,1,0], [-1,-1,0], [-2,-1,4], [2,1,4]]
#for hkl in hkls:
#    XRD.simulation.tracking_specific_reflections(phase = "Hexagonal", detector = "Rigaku", sample_detector_distance = 60.87, tilting_angle = 0,wavelength = 0.7107, rot_x_start = -180, rot_x_end = 180, step_rot_x = 3, rot_z_start = -180, rot_z_end = 180, step_rot_z=3, E_bandwidth = 5, desired_reflections_list = [hkl], margin = 0)