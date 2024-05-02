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
    tilting_angle = np.radians(tilting_angle)
    ki = np.array([2*np.pi/wavelength, 0, 0])
    kf = Q_hkl + ki
    
    dx, dy, dz = 0, 0, 0
    
    kfxy = kf.copy()
    kfxy[2] = 0
    
    kfxz = kf.copy()
    kfxz[1] = 0
    
    if kf[0] > 0:
        dx = 1
    
    try:
        denominator = np.dot(kfxz, ki) / (np.linalg.norm(kfxz) * np.linalg.norm(ki))
        if abs(denominator) < 1:
            numerator = np.cos(tilting_angle) / np.tan(np.arccos(denominator))
            dz = np.sign(kf[2]) * sample_detector_distance / (numerator + np.sin(tilting_angle))

    except ZeroDivisionError:
        pass
    
    try:
        dy = np.sign(kf[1])*(sample_detector_distance - dz*np.sin(tilting_angle))*np.tan(np.arccos(np.dot(kfxy, ki) / (np.linalg.norm(kfxy) * np.linalg.norm(ki))))
    except ZeroDivisionError:
        pass

    return dx, dy, dz

def diffraction_in_detector(dx, dy, dz, detector):
    #Detector must be an object
    if dx > 0:
        if (dz <= detector.Max_Detectable_Z() and dz >= detector.Min_Detectable_Z()) and (dy <= detector.Max_Detectable_Y() and dy >= detector.Min_Detectable_Y()):
            return True
        else:
            return False

    else:
        return False

"""
def diffraction_direction(Q_hkls, wavelength, sample_detector_distance, tilting_angle, detector):
    tilting_angle = np.radians(tilting_angle)
    ki = np.array([2*np.pi/wavelength, 0, 0])
    kf = Q_hkls + ki[:, np.newaxis]  # Add ki to each Q_hkl
    
    dx, dy, dz = np.zeros_like(Q_hkls), np.zeros_like(Q_hkls), np.zeros_like(Q_hkls)

    kfxy = kf.copy()
    kfxy[:, 2] = 0  # Set the third component to zero for all rows

    kfxz = kf.copy()
    kfxz[:, 1] = 0  # Set the second component to zero for all rows

    # Calculate dx
    dx[kf[:, 0] > 0] = 1

    try:
        denominator = np.dot(kfxz, ki) / (np.linalg.norm(kfxz, axis=1) * np.linalg.norm(ki))
        mask = np.abs(denominator) < 1
        numerator = np.cos(tilting_angle) / np.tan(np.arccos(denominator[mask]))
        dz[mask] = np.sign(kf[mask, 2]) * sample_detector_distance / (numerator + np.sin(tilting_angle))

    except ZeroDivisionError:
        pass

    try:
        mask = np.abs(np.dot(kfxy, ki) / (np.linalg.norm(kfxy, axis=1) * np.linalg.norm(ki))) < 1
        dy[mask] = np.sign(kf[mask, 1]) * (sample_detector_distance - dz[mask] * np.sin(tilting_angle)) * np.tan(
            np.arccos(np.dot(kfxy[mask], ki) / (np.linalg.norm(kfxy[mask], axis=1) * np.linalg.norm(ki))))
    except ZeroDivisionError:
        pass

    # Use diffraction_in_detector function with arrays and create a mask
    detection_mask = diffraction_in_detector(dx, dy, dz, detector)

    # Apply the mask to dx, dy, and dz
    dx = dx[detection_mask]
    dy = dy[detection_mask]
    dz = dz[detection_mask]

    return dx, dy, dz
"""

    

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

def construct_kf_exp(wavelength, y_distance_from_center, z_distance_from_center, sample_detector_distance, tilting_angle = 0):

    """This function is dedicated to single crystal orientation from exp data"""

    tilting_angle = np.deg2rad(tilting_angle)

    ki = 2*np.pi/wavelength

    kfy = (ki**2)*(1 - (1/(1/((sample_detector_distance/(z_distance_from_center*np.cos(tilting_angle)) - np.tan(tilting_angle))**2 + 1) + (np.cos(np.arctan(np.cos(tilting_angle)/(sample_detector_distance/z_distance_from_center - np.sin(tilting_angle)))
)/np.cos(np.arctan(y_distance_from_center/(sample_detector_distance - z_distance_from_center*np.sin(tilting_angle)))))**2)))

    kfz = (ki**2 - kfy**2)/((sample_detector_distance/(z_distance_from_center*np.cos(tilting_angle)) - np.tan(tilting_angle))**2 + 1)

    kfx = ki**2 - (kfy**2 + kfz**2)
    
    return np.array([kfx, kfy, kfz])
#"""
def single_crystal_orientation(phase, wavelength, sample_detector_distance, 
                               rotations_peak_1, y_distance_from_center_peak_1, z_distance_from_center_peak_1, hkl_peak_1, 
                               rotations_peak_2, y_distance_from_center_peak_2, z_distance_from_center_peak_2, hkl_peak_2, 
                               rotations_peak_3, y_distance_from_center_peak_3, z_distance_from_center_peak_3, hkl_peak_3, 
                               crystal_orient_guess, tilting_angle = 0, rotation_order = "xyz"):

    #This long function is to be used when trying to retreive single crystal orientation given 3 Braggs.

    if phase == "Hexagonal":
        lattice = Hexagonal_Lattice()
        bounds = ([-lattice.c] * 9, [lattice.c] * 9)
    
    elif phase == "Monoclinic":
        lattice = Monoclinic_Lattice()
        bounds = ([-lattice.a] * 9, [lattice.a] * 9)

    a = lattice.a
    b = lattice.b
    c = lattice.c
    alpha = lattice.alpha
    beta = lattice.beta
    gamma = lattice.gamma

    kf1 = construct_kf_exp(wavelength, y_distance_from_center_peak_1, z_distance_from_center_peak_1, sample_detector_distance = sample_detector_distance, tilting_angle = tilting_angle)
    kf2 = construct_kf_exp(wavelength, y_distance_from_center_peak_2, z_distance_from_center_peak_2, sample_detector_distance = sample_detector_distance, tilting_angle = tilting_angle)
    kf3 = construct_kf_exp(wavelength, y_distance_from_center_peak_3, z_distance_from_center_peak_3, sample_detector_distance = sample_detector_distance, tilting_angle = tilting_angle)
    #kf4 = Construct_kf_exp(wavelength = wavelength, y_distance_from_center = y_distance_from_center_peak_4, z_distance_from_center = z_distance_from_center_peak_4, sample_detector_distance = sample_detector_distance, tilting_angle = tilting_angle)

    ki = 2*np.pi/wavelength #np.array([2*np.pi/wavelength, 0, 0])**2

    def residuals(params):
        A, B, C, D, E, F, G, H, I = params
        Cryst_Orient = np.array([
            [A, B, C],
            [D, E, F],
            [G, H, I],
        ])
            
        Cryst_Orient_1 = apply_rotation(initial_matrix = Cryst_Orient, rotx = rotations_peak_1[0], roty = rotations_peak_1[1], rotz = rotations_peak_1[2], rotation_order=rotation_order)
        Cryst_Orient_2 = apply_rotation(initial_matrix = Cryst_Orient, rotx = rotations_peak_2[0], roty = rotations_peak_2[1], rotz = rotations_peak_2[2], rotation_order=rotation_order)
        Cryst_Orient_3 = apply_rotation(initial_matrix = Cryst_Orient, rotx = rotations_peak_3[0], roty = rotations_peak_3[1], rotz = rotations_peak_3[2], rotation_order=rotation_order)
        #Cryst_Orient_4 = XRD.utils.apply_rotation(initial_matrix = Cryst_Orient, rotx = rotations_peak_4[0], roty = rotations_peak_4[1], rotz = rotations_peak_4[2])

        reciprocal_lattice_1 = cal_reciprocal_lattice(Cryst_Orient_1)
        reciprocal_lattice_2 = cal_reciprocal_lattice(Cryst_Orient_2)
        reciprocal_lattice_3 = cal_reciprocal_lattice(Cryst_Orient_3)
        #a_rec_peak_4, b_rec_peak_4, c_rec_peak_4 = XRD.utils.cal_reciprocal_lattice(Cryst_Orient_4[0], Cryst_Orient_4[1], Cryst_Orient_4[2])

        GG_peak_1 = calculate_Q_hkl(hkl_peak_1, reciprocal_lattice_1)
        GG_peak_2 = calculate_Q_hkl(hkl_peak_2, reciprocal_lattice_2)
        GG_peak_3 = calculate_Q_hkl(hkl_peak_3, reciprocal_lattice_3)
        #GG_peak_4 = XRD.utils.calculate_Q_hkl(hkl_peak_4, a_rec_peak_4, b_rec_peak_4, c_rec_peak_4)

        # Constraints
        constraint1 = A**2 + B**2 + C**2 - a**2
        constraint2 = D**2 + E**2 + F**2 - b**2
        constraint3 = G**2 + H**2 + I**2 - c**2
        constraint4 =  np.dot(Cryst_Orient[0],  Cryst_Orient[1])   - a*b*np.cos(np.deg2rad(gamma))
        constraint5 =  np.dot(Cryst_Orient[1],  Cryst_Orient[2])   - b*c*np.cos(np.deg2rad(alpha))
        constraint6 =  np.dot(Cryst_Orient[2],  Cryst_Orient[0])   - c*a*np.cos(np.deg2rad(beta ))
        constraint7 = np.dot(Cryst_Orient_1[0], Cryst_Orient_1[1]) - a*b*np.cos(np.deg2rad(gamma))
        constraint8 = np.dot(Cryst_Orient_1[1], Cryst_Orient_1[2]) - b*c*np.cos(np.deg2rad(alpha))
        constraint9 = np.dot(Cryst_Orient_1[2], Cryst_Orient_1[0]) - c*a*np.cos(np.deg2rad(beta ))
        #constraint10 = np.dot(Cryst_Orient_2[0], Cryst_Orient_2[1]) - a*b*np.cos(np.deg2rad(gamma))
        #constraint11 = np.dot(Cryst_Orient_2[1], Cryst_Orient_2[2]) - b*c*np.cos(np.deg2rad(alpha))
        #constraint12 = np.dot(Cryst_Orient_2[2], Cryst_Orient_2[0]) - c*a*np.cos(np.deg2rad(beta ))
        #constraint13 = np.dot(Cryst_Orient_3[0], Cryst_Orient_3[1]) - a*b*np.cos(np.deg2rad(gamma))
        #constraint14 = np.dot(Cryst_Orient_3[1], Cryst_Orient_3[2]) - b*c*np.cos(np.deg2rad(alpha))
        #constraint15 = np.dot(Cryst_Orient_3[2], Cryst_Orient_3[0]) - c*a*np.cos(np.deg2rad(beta ))
        

        res_1 = (GG_peak_1[0] + ki)**2 - kf1[0]
        res_2 =  GG_peak_1[1]**2 - kf1[1]
        res_3 =  GG_peak_1[2]**2 - kf1[2]
        res1 = [res_1,res_2,res_3]

        res_4 = (GG_peak_2[0] + ki)**2 - kf2[0]
        res_5 =  GG_peak_2[1]**2 - kf2[1]
        res_6 =  GG_peak_2[2]**2 - kf2[2]
        res2 = [res_4,res_5,res_6]

        res_7 = (GG_peak_3[0] + ki)**2 - kf3[0]
        res_8 =  GG_peak_3[1]**2 - kf3[1]
        res_9 =  GG_peak_3[2]**2 - kf3[2]
        res3 = [res_7, res_8, res_9]

        return  np.concatenate((res1, res2, res3, [constraint1, constraint2, constraint3, constraint4, constraint5, constraint6, constraint7, constraint8, constraint9]))

        #return [res_1, res_2, res_3, res_4, res_5, res_6, res_7, res_8, res_9, constraint1, constraint2, constraint3,constraint4, constraint5, constraint6, constraint7, constraint8, constraint9, constraint10, constraint11, constraint12, constraint13, constraint14, constraint15]

    sol = least_squares(residuals, crystal_orient_guess, bounds=bounds, verbose = 2)
    print(sol.jac)
    #np.linalg.inv()

    solution = np.array([
        [sol.x[0], sol.x[1], sol.x[2]],
        [sol.x[3], sol.x[4], sol.x[5]],
        [sol.x[6], sol.x[7], sol.x[8]]
    ])

    #current_crystal_orientation = apply_rotation(solution, rotx = rotations_peak_3[0], roty = rotations_peak_3[1], rotz = rotations_peak_3[2], rotation_order = "xyz")

    
    return np.round(solution, 3)
#"""
    



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
        #self.tilting_angle = self.tilting_angle
        #max_theta_z = np.arctan(np.cos(self.tilting_angle)/((self.sample_detector_distance/(self.height - self.beam_center[1])) - np.sin(self.tilting_angle)))
        #max_dz = (1 - self.margin/100)*self.sample_detector_distance*np.tan(max_theta_z)/(np.cos(self.tilting_angle) + np.tan(max_theta_z)*np.sin(self.tilting_angle))

        max_dz = (1 - self.margin/100)*(self.height - self.beam_center[1])*np.cos(self.tilting_angle)

        return max_dz
    
    def Min_Detectable_Z(self):
        #self.tilting_angle = self.tilting_angle
        #max_theta_z = np.arctan(np.cos(self.tilting_angle)/((self.sample_detector_distance/self.height) - np.sin(self.tilting_angle)))

        #min_dz = self.sample_detector_distance*np.tan(max_theta_z)/(np.cos(self.tilting_angle) + np.tan(max_theta_z)*np.sin(self.tilting_angle))

        min_dz = (-self.beam_center[1])*(1 - self.margin/100)

        return min_dz
    
    def Max_Detectable_Y(self):
        #max_theta_y = np.arctan((self.width/2 - self.beam_center[0])/(self.sample_detector_distance))
        #max_dy = (1 - self.margin/100)*self.sample_detector_distance*np.tan(max_theta_y)
        max_dy = (1 - self.margin/100)*(self.width/2 - self.beam_center[0])
        return max_dy

    def Min_Detectable_Y(self):
        #min_theta_y = np.arctan((-self.width/2 - self.beam_center[0])/(self.sample_detector_distance))
        #min_dy = (1 - self.margin/100)*self.sample_detector_distance*np.tan(min_theta_y)
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



#hkls = [[1,1,0], [-1,-1,0], [-2,-1,4], [2,1,4]]
#for hkl in hkls:
#    XRD.simulation.tracking_specific_reflections(phase = "Hexagonal", detector = "Rigaku", sample_detector_distance = 60.87, tilting_angle = 0,wavelength = 0.7107, rot_x_start = -180, rot_x_end = 180, step_rot_x = 3, rot_z_start = -180, rot_z_end = 180, step_rot_z=3, E_bandwidth = 5, desired_reflections_list = [hkl], margin = 0)




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

    #total_rotation = np.dot(rotation_matrix_z, np.dot(rotation_matrix_y, rotation_matrix_x))
    #rotated_matrix = np.dot(total_rotation,initial_matrix.T).T

    total_rotation = np.dot(rotation_matrix_x, np.dot(rotation_matrix_y, rotation_matrix_z))
    rotated_matrix = np.dot(initial_matrix, total_rotation)

    return rotated_matrix

"""

"""
def cal_reciprocal_lattice(vector1, vector2, vector3):
    volume = np.dot(vector1, np.cross(vector2, vector3))
    rec_vec1 = (2*np.pi/volume)*np.cross(vector2, vector3)
    rec_vec2 = (2*np.pi/volume)*np.cross(vector3, vector1)
    rec_vec3 = (2*np.pi/volume)*np.cross(vector1, vector2)
    return rec_vec1, rec_vec2, rec_vec3

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

"""
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
"""