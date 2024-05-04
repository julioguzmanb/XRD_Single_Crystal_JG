import numpy as np
import plotly.graph_objs as go
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares


def energy_to_wavelength(energy):
    """Energy in KeV. Output will be in Å """
    h = 4.135667696e-15 #eV-s
    c = 299792458 #m/s
    wavelength = (h/1000)*(c*1e10)/energy # Å

    return wavelength

def wavelength_to_energy(wavelength):
    """wavelength in Å. Output will be in KeV """
    h = 4.135667696e-15 #eV-s
    c = 299792458 #m/s
    energy = (h/1000)*(c*1e10)/wavelength # KeV

    return energy

def apply_rotation(initial_matrix, rotx=0, roty=0, rotz=0, rotation_order = "xyz"):
    rotation_quaternion_x = R.from_euler('x', -rotx, degrees=True).as_quat()
    rotation_quaternion_y = R.from_euler('y', -roty, degrees=True).as_quat()
    rotation_quaternion_z = R.from_euler('z', -rotz, degrees=True).as_quat()

    if rotation_order == "xyz":
        total_rotation_quaternion = R.from_quat(rotation_quaternion_x) * R.from_quat(rotation_quaternion_y) * R.from_quat(rotation_quaternion_z)

    elif rotation_order == "xzy":
        total_rotation_quaternion = R.from_quat(rotation_quaternion_x) * R.from_quat(rotation_quaternion_z) * R.from_quat(rotation_quaternion_y)
    
    elif rotation_order == "yxz":
        total_rotation_quaternion = R.from_quat(rotation_quaternion_y) * R.from_quat(rotation_quaternion_x) * R.from_quat(rotation_quaternion_z)

    elif rotation_order == "yzx":
        total_rotation_quaternion = R.from_quat(rotation_quaternion_y) * R.from_quat(rotation_quaternion_z) * R.from_quat(rotation_quaternion_x)
    
    elif rotation_order == "zxy":
        total_rotation_quaternion = R.from_quat(rotation_quaternion_z) * R.from_quat(rotation_quaternion_x) * R.from_quat(rotation_quaternion_y)
    
    elif rotation_order == "zyx":
        total_rotation_quaternion = R.from_quat(rotation_quaternion_z) * R.from_quat(rotation_quaternion_y) * R.from_quat(rotation_quaternion_x)

    total_rotation_matrix = total_rotation_quaternion.as_matrix()

    rotated_matrix = np.round(np.dot(initial_matrix, total_rotation_matrix), 5)
    
    return rotated_matrix


def allowed_reflections(phase, hkl):
    h, k, l = hkl
    
    if (h == k == l == 0):
        return False  # Ruling out the [0,0,0]
    
    if phase == "Hexagonal":
        # Rules for space group 167
        conditions = [
            ((-h + k + l) % 3 == 0) and (h or k != 0) and (l % 2 == 0),
            ((h - k + l) % 3 == 0) and (h or k != 0) and (l % 2 == 0),
            (h == k != l) and (l % 3 == 0) and (h == k != 0),
            (h == -k) and ((h + l) % 3 == 0) and (h or k != 0),
            (h == -k) and ((k + l) % 3 == 0) and (h or k != 0),
            (h == k == 0) and (l % 2 == 0) and (l % 3 == 0)
        ]
        
        if any(conditions):
            return True
    
    return False

def create_possible_reflections(phase, smallest_number, largest_number):
    rango_hkl = np.arange(smallest_number, largest_number + 1)
    h, k, l = np.meshgrid(rango_hkl, rango_hkl, rango_hkl)
    
    # Create the combinations
    combinaciones_hkl = np.column_stack((h.ravel(), k.ravel(), l.ravel()))
    
    # Filter out combinations based on allowed_reflections
    allowed_mask = np.apply_along_axis(lambda x: allowed_reflections(phase, x), axis=1, arr=combinaciones_hkl)

    combinaciones_hkl = combinaciones_hkl[allowed_mask]

    return combinaciones_hkl

def cal_reciprocal_lattice(lattice):

    reciprocal_lattice = np.linalg.inv(lattice).T*2*np.pi

    return reciprocal_lattice

def calculate_Q_hkl(hkl, reciprocal_lattice):

    Q_hkl = np.dot(hkl,reciprocal_lattice)

    return Q_hkl


def check_Bragg_condition(Q_hkls, wavelength, E_bandwidth):

    ewald_sphere = Ewald_Sphere(wavelength, E_bandwidth)

    ki = np.array([2*np.pi/wavelength, 0, 0]).reshape(1, -1)
    kf = Q_hkls + ki  # Add ki to each Q_hkl

    kf_hkl_module = np.linalg.norm(kf,axis = 1)

    in_bragg_condition = (kf_hkl_module >= ewald_sphere.Get_Inner_Radius()) & (kf_hkl_module <= ewald_sphere.Get_Outer_Radius())
        
 
    return in_bragg_condition



def diffraction_direction(Q_hkls, wavelength, sample_detector_distance, tilting_angle):
    tilting_angle = np.radians(tilting_angle)

    ki = np.array([2*np.pi/wavelength, 0, 0]).reshape(1, -1)
    kf_hkls = Q_hkls + ki 
    
    dx, dy, dz = np.zeros(len(kf_hkls)), np.zeros(len(kf_hkls)), np.zeros(len(kf_hkls))

    kfxy = kf_hkls.copy()
    kfxy[:, 2] = 0  # Set the third component to zero for all rows

    kfxz = kf_hkls.copy()
    kfxz[:, 1] = 0  # Set the second component to zero for all rows

    # Calculate dx
    dx[kf_hkls[:, 0] > 0] = 1

    try:
        denominator = kfxz[:,0]/np.linalg.norm(kfxz, axis=1)
        mask = np.abs(denominator) < 1
        numerator = np.cos(tilting_angle) / np.tan(np.arccos(denominator[mask]))
        dz[mask] = np.sign(kf_hkls[mask, 2]) * sample_detector_distance / (numerator + np.sin(tilting_angle))

    except ZeroDivisionError:
        pass

    try:
        mask = np.abs(kfxz[:,0]/(np.linalg.norm(kfxy, axis=1))) < 1
        dy[mask] = np.sign(kf_hkls[mask, 1]) * (sample_detector_distance - dz[mask] * np.sin(tilting_angle)) * np.tan(np.arccos(kfxz[:,0][mask] / (np.linalg.norm(kfxy[mask], axis=1))))
    except ZeroDivisionError:
        pass

    diffracted_information = np.stack((dx, dy, dz), axis=1)

    return diffracted_information

def diffraction_in_detector(diffracted_information, detector):
    #Detector must be an object
    mask = (diffracted_information[:,0] > 0) & (diffracted_information[:,1] <= detector.Max_Detectable_Y()) & (diffracted_information[:,1] >= detector.Min_Detectable_Y()) & (diffracted_information[:,2] <= detector.Max_Detectable_Z()) & (diffracted_information[:,2] >= detector.Min_Detectable_Z())

    return mask


def calculate_two_theta_angle(phase, hkl, wavelength):

    if phase == "Monoclinic":
        lattice_structure = Monoclinic_Lattice()
    
    elif phase == "Hexagonal":
        lattice_structure = Hexagonal_Lattice()


    Q = calculate_Q_hkl(hkl, lattice_structure.reciprocal_lattice)

    Q_norm = np.linalg.norm(Q)
    two_theta = np.rad2deg(2*np.arcsin(Q_norm*wavelength/(4*np.pi)))

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

    Q_hkl_1 = calculate_Q_hkl(hkl_1, lattice_structure.reciprocal_lattice)
    Q_hkl_2 = calculate_Q_hkl(hkl_2, lattice_structure.reciprocal_lattice)
    
    angle = np.rad2deg(np.arccos(np.dot(Q_hkl_1, Q_hkl_2)/(np.linalg.norm(Q_hkl_1)*np.linalg.norm(Q_hkl_2))))

    return angle

def single_crystal_orientation(phase, wavelength, sample_detector_distance, beam_center,
                               hkls, rotations, y_coordinates, z_coordinates,
                               crystal_orient_guess, tilting_angle = 0, rotation_order = "xyz"):

    #This long function is to be used when trying to retreive single crystal orientation given 3 Braggs.
    y_distances = np.array(y_coordinates) - beam_center[0]
    z_distances = np.array(z_coordinates) - beam_center[1]

    def construct_kf_exp(wavelength, y_distance_from_center, z_distance_from_center, sample_detector_distance, tilting_angle = 0):

        tilting_angle = np.deg2rad(tilting_angle)
        ki = 2*np.pi/wavelength
        kfy = (ki**2)*(1 - (1/(1/((sample_detector_distance/(z_distance_from_center*np.cos(tilting_angle)) - np.tan(tilting_angle))**2 + 1) + (np.cos(np.arctan(np.cos(tilting_angle)/(sample_detector_distance/z_distance_from_center - np.sin(tilting_angle))))/np.cos(np.arctan(y_distance_from_center/(sample_detector_distance - z_distance_from_center*np.sin(tilting_angle)))))**2)))
        kfz = (ki**2 - kfy**2)/((sample_detector_distance/(z_distance_from_center*np.cos(tilting_angle)) - np.tan(tilting_angle))**2 + 1)
        kfx = ki**2 - (kfy**2 + kfz**2)
        return np.array([kfx, kfy, kfz])
    
    if phase == "Hexagonal":
        lattice = Hexagonal_Lattice()
        #bounds = ([-lattice.c] * 9, [lattice.c] * 9)
    
    elif phase == "Monoclinic":
        lattice = Monoclinic_Lattice()
        #bounds = ([-lattice.a] * 9, [lattice.a] * 9)

    a, b, c, alpha, beta, gamma = lattice.a, lattice.b, lattice.c, lattice.alpha, lattice.beta, lattice.gamma

    ki = 2*np.pi/wavelength 

    def residuals(params):
        A, B, C, D, E, F, G, H, I = params
        Cryst_Orient = np.array([[A, B, C], [D, E, F], [G, H, I]])
        
        ress = []

        for i in range(len(hkls)):
            kf = construct_kf_exp(wavelength, y_distances[i], z_distances[i], sample_detector_distance = sample_detector_distance, tilting_angle = tilting_angle)

            Cryst_Orientation = apply_rotation(initial_matrix = Cryst_Orient, rotx = rotations[i][0], roty = rotations[i][1], rotz = rotations[i][2], rotation_order=rotation_order)
            reciprocal_lattice = cal_reciprocal_lattice(Cryst_Orientation)
            GG_peak = calculate_Q_hkl(hkls[i], reciprocal_lattice)

            res_1 = (GG_peak[0] + ki)**2 - kf[0]
            res_2 =  GG_peak[1]**2       - kf[1]
            res_3 =  GG_peak[2]**2       - kf[2]

            ress.append(res_1)
            ress.append(res_2)
            ress.append(res_3)

        # Constraints
        constraint1 = A**2 + B**2 + C**2 - a**2
        constraint2 = D**2 + E**2 + F**2 - b**2
        constraint3 = G**2 + H**2 + I**2 - c**2
        constraint4 =  np.dot(Cryst_Orient[0],  Cryst_Orient[1])   - a*b*np.cos(np.deg2rad(gamma))
        constraint5 =  np.dot(Cryst_Orient[1],  Cryst_Orient[2])   - b*c*np.cos(np.deg2rad(alpha))
        constraint6 =  np.dot(Cryst_Orient[2],  Cryst_Orient[0])   - c*a*np.cos(np.deg2rad(beta ))

        return  np.concatenate((ress, [constraint1, constraint2, constraint3, constraint4, constraint5, constraint6]))


    #sol = least_squares(residuals, crystal_orient_guess, bounds=bounds, verbose = 2)
    sol = least_squares(residuals, crystal_orient_guess, verbose = 2, method = "lm")
    #print(sol.jac)
    #np.linalg.inv()

    print(sol)

    solution = np.array([
        [sol.x[0], sol.x[1], sol.x[2]],
        [sol.x[3], sol.x[4], sol.x[5]],
        [sol.x[6], sol.x[7], sol.x[8]]
    ])


    print(solution)
    return np.round(solution, 3)



class Lattice_Structure:
    def __init__(self, a, b, c, alpha, beta, gamma, initial_crystal_orientation, Vanadium_fractional_position, Oxygen_fractional_position):
        self.a = a
        self.b = b
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.initial_crystal_orientation = initial_crystal_orientation
        self.Vanadium_fractional_position = Vanadium_fractional_position
        self.Oxygen_fractional_position = Oxygen_fractional_position
        self.crystal_orientation = initial_crystal_orientation
        self.reciprocal_lattice = cal_reciprocal_lattice(self.crystal_orientation)
    
    def Apply_Rotation(self, rotx = 0, roty = 0, rotz = 0, rotation_order = "xyz"):
        self.crystal_orientation = np.round(apply_rotation(self.initial_crystal_orientation, rotx, roty, rotz, rotation_order=rotation_order),5)
        self.reciprocal_lattice = cal_reciprocal_lattice(self.crystal_orientation)
    
        #self.crystal_orientation = apply_rotation(self.crystal_orientation, rotx, roty, rotz)
    
class Hexagonal_Lattice(Lattice_Structure):
    def __init__(self, initial_crystal_orientation = None):
        a = 4.9525
        b = 4.9525
        c = 14.00093
        alpha = 90
        beta = 90
        gamma = 120
        V = np.sin(np.deg2rad(60))*a*b*c
        if initial_crystal_orientation is None:
            initial_crystal_orientation = np.array([
            [a     , 0               , 0  ],
            [-0.5*b, np.sqrt(3)*b/2, 0  ],
            [0       , 0               , c]
            ])

        Vanadium_fractional_position = np.array([0.00000,    0.00000,    0.34670])
        Oxygen_fractional_position   = np.array([0.31480,    0.00000,    0.25000])

        super().__init__(a, b, c, alpha, beta, gamma, initial_crystal_orientation, Vanadium_fractional_position, Oxygen_fractional_position)
        
class Monoclinic_Lattice(Lattice_Structure):
    def __init__(self, initial_crystal_orientation = None):
        a = 7.266
        b = 5.0024
        c = 5.5479
        alpha = 90
        beta = 96.760
        gamma = 90
        V_M = a*b*c*np.sin(np.deg2rad(beta))


        if initial_crystal_orientation is None:
            initial_crystal_orientation = np.array([
            [b/2 - np.sqrt((c**2 + a*c*np.cos(np.radians(beta)) - (b**2)/4)/3), 
            b/np.sqrt(12) - np.sqrt(c**2 + a*c*np.cos(np.radians(beta)) - (b**2)/4),  
            np.sqrt((14*c**2 - 4*a**2 + a*c*np.cos(np.radians(beta)))/9)],

            [b,0,0],

            [b/4 - np.sqrt((c**2 + a*c*np.cos(np.radians(beta)) - (b**2)/4)/12),
            b/np.sqrt(48) - 0.5*np.sqrt(c**2 + a*c*np.cos(np.radians(beta)) - (b**2)/4), 
            -np.sqrt((14*c**2 - 4*a**2 + a*c*np.cos(np.radians(beta)))/9)]
            ])

        Vanadium_fractional_position = np.array([0.34460,    0.00500,    0.29850])
        Oxygen_fractional_position   = np.array([[0.41100,    0.84700,    0.65000], [0.25000,    0.18000,    0.00000]])

        super().__init__(a, b, c, alpha, beta, gamma, initial_crystal_orientation, Vanadium_fractional_position, Oxygen_fractional_position)

class Detector:
    def __init__(self, detector_type, sample_detector_distance = None, tilting_angle = 0, beam_center = (0,0), margin = 0):

        #The beam_center = (0,0) corresponds to the lower center part of the detector. This is to be changed eventually...

        self.detector_type = detector_type #I just want to make it accessible 

        self.beam_center = beam_center

        self.margin = margin

        self.tilting_angle = np.radians(tilting_angle)

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
        
        
        
    def Max_Detectable_Z(self):
        max_dz = (1 - self.margin/100)*(self.height - self.beam_center[1])*np.cos(self.tilting_angle)

        return max_dz
    
    def Min_Detectable_Z(self):
        min_dz = (-self.beam_center[1])*(1 - self.margin/100)

        return min_dz
    
    def Max_Detectable_Y(self):
        max_dy = (1 - self.margin/100)*(self.width/2 - self.beam_center[0])
        return max_dy

    def Min_Detectable_Y(self):
        min_dy = (1 - self.margin/100)*(-self.width/2 - self.beam_center[0])
        return min_dy
        
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


""" #I am still thinking whether is better to use quaternions or not...
def apply_rotation(initial_matrix, rotx = 0, roty = 0, rotz = 0):
    rotx = np.radians(rotx)
    roty = np.radians(roty)
    rotz = np.radians(rotz)
    rotation_matrix_x = np.array([
        [1, 0, 0],
        [0, np.cos(rotx), -np.sin(rotx)],
        [0, np.sin(rotx),  np.cos(rotx)]
    ])

    rotation_matrix_y = np.array([
        [np.cos(roty), 0, np.sin(roty)],
        [0, 1, 0],
        [-np.sin(roty), 0, np.cos(roty)]
    ])

    rotation_matrix_z = np.array([
        [np.cos(rotz), -np.sin(rotz), 0],
        [np.sin(rotz),  np.cos(rotz), 0],
        [0, 0, 1]
    ])

    total_rotation = np.dot(rotation_matrix_x, np.dot(rotation_matrix_y, rotation_matrix_z))
    rotated_matrix = np.dot(initial_matrix, total_rotation)

    return rotated_matrix

"""


"""
def allowed_reflections(phase, hkl):
    h, k, l = hkl
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
"""


