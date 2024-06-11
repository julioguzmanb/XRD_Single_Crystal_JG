import numpy as np
import plotly.graph_objs as go
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
import pyFAI.detectors

def energy_to_wavelength(energy):
    """
    Convert energy (in electron volts) to wavelength (in meters).

    Parameters:
    energy (float): The energy in electron volts (eV).

    Returns:
    float: The corresponding wavelength in meters (m).
    """
    h = 4.135667696e-15  # Planck's constant in eV·s
    c = 299792458  # Speed of light in m/s
    wavelength = h * c / energy  # Calculate wavelength in meters

    return wavelength

def wavelength_to_energy(wavelength):
    """
    Convert wavelength (in meters) to energy (in electron volts).

    Parameters:
    wavelength (float): The wavelength in meters (m).

    Returns:
    float: The corresponding energy in electron volts (eV).
    """
    h = 4.135667696e-15  # Planck's constant in eV·s
    c = 299792458  # Speed of light in m/s
    energy = h * c / wavelength  # Calculate energy in electron volts

    return energy

def apply_rotation(initial_matrix, rotx=0, roty=0, rotz=0, rotation_order="xyz"):
    """
    Apply a series of rotations to an initial matrix using specified Euler angles and rotation order.

    Parameters:
    initial_matrix (numpy.ndarray): The matrix to be rotated.
    rotx (float): Rotation angle around the x-axis in degrees. Default is 0.
    roty (float): Rotation angle around the y-axis in degrees. Default is 0.
    rotz (float): Rotation angle around the z-axis in degrees. Default is 0.
    rotation_order (str): The order of rotations, specified as a string of axes (e.g., "xyz", "zyx"). Default is "xyz".

    Returns:
    numpy.ndarray: The rotated matrix.
    """
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
    """
    Determine if a given set of Miller indices (hkl) is allowed for a specific crystal phase.

    Parameters:
    phase (str): The crystal phase, either "Hexagonal" or "Monoclinic".
    hkl (tuple of int): A tuple containing the Miller indices (h, k, l).

    Returns:
    bool: True if the reflection is allowed for the specified phase, False otherwise.
    """
    h, k, l = hkl
    
    if (h == k == l == 0):
        return False  # Ruling out the [0,0,0]
    
    if phase == "Hexagonal":
        # Rules for space group 167
        if ((-h + k + l)%3 == 0) and (h != 0) and (k != 0) and (l != 0):
            return True
        
        elif (h == 0) and (l%2 == 0) and ((k + l)%3 == 0) and (k != 0) and (l != 0):
            return True

        elif (k == 0) and (l%2 == 0) and ((h - l)%3 == 0) and (h != 0) and (l != 0):
            return True

        elif (l == 0) and ((h - k)%3 == 0) and (h != 0) and (k != 0):
            return True
        
        elif (h == k != 0) and (l%3 == 0):
            return True

        elif (k == 0) and (l == 0) and (h%3 == 0) and (h != 0):
            return True
        
        elif (h == 0) and (l == 0) and (k%3 == 0) and (k != 0):
            return True
        
        elif (h == 0) and (k == 0) and (l%6 == 0) and (l != 0):
            return True
        
        
    elif phase == "Monoclinic":

        if ((h + k)%2 == 0) and (h != 0) and (k != 0) and (l != 0):
            return True
        elif (h == 0) and (k%2 == 0) and (k != 0) and (l != 0):
            return True
        elif (k == 0) and (h%2 == 0) and (l%2 == 0) and (h != 0) and (l != 0):
            return True
        elif (l == 0) and ((h + k)%2 == 0) and (h != 0) and (k != 0):
            return True
        elif (k == l == 0) and (h%2 == 0):
            return True
        elif (h == l == 0) and (k%2 == 0):
            return True
        elif (h == k == 0) and (l%2 == 0):
            return True
        elif (h == 2) and (k == 2) and (l == 1): #Magnetic one
            return True
        elif (h == -2) and (k == 2) and (l == -1): #Magnetic one
            return True
        elif (h == 2) and (k == -2) and (l == 1): #Magnetic one
            return True
        elif (h == -2) and (k == -2) and (l == -1): #Magnetic one
            return True
        
    return False

def create_possible_reflections(phase, smallest_number, largest_number):
    """
    Generate all possible reflections within a specified range for a given crystal phase.

    Parameters:
    phase (str): The crystal phase, e.g., "Hexagonal" or "Monoclinic".
    smallest_number (int): The smallest Miller index to consider.
    largest_number (int): The largest Miller index to consider.

    Returns:
    numpy.ndarray: An array of allowed (h, k, l) reflections.
    """
    rango_hkl = np.arange(smallest_number, largest_number + 1)
    h, k, l = np.meshgrid(rango_hkl, rango_hkl, rango_hkl)
    
    # Create the combinations
    combinaciones_hkl = np.column_stack((h.ravel(), k.ravel(), l.ravel()))
    
    # Filter out combinations based on allowed_reflections
    allowed_mask = np.apply_along_axis(lambda x: allowed_reflections(phase, x), axis=1, arr=combinaciones_hkl)

    combinaciones_hkl = combinaciones_hkl[allowed_mask]

    return combinaciones_hkl

def cal_reciprocal_lattice(lattice):
    """
    Calculate the reciprocal lattice vectors from the direct lattice vectors.

    Parameters:
    lattice (numpy.ndarray): A 3x3 matrix representing the direct lattice vectors.

    Returns:
    numpy.ndarray: A 3x3 matrix representing the reciprocal lattice vectors.
    """
    reciprocal_lattice = np.linalg.inv(lattice).T * 2 * np.pi

    return reciprocal_lattice

def calculate_Q_hkl(hkl, reciprocal_lattice):
    """
    Calculate the Q vector for given Miller indices and reciprocal lattice.

    Parameters:
    hkl (numpy.ndarray): An array of Miller indices (h, k, l).
    reciprocal_lattice (numpy.ndarray): A 3x3 matrix representing the reciprocal lattice vectors.

    Returns:
    numpy.ndarray: The Q vector corresponding to the given Miller indices.
    """
    Q_hkl = np.dot(hkl, reciprocal_lattice)

    return Q_hkl

def calculate_dspacing(hkl, reciprocal_lattice):
    """
    Calculate the d-spacing for given Miller indices and reciprocal lattice.

    Parameters:
    hkl (numpy.ndarray): An array of Miller indices (h, k, l).
    reciprocal_lattice (numpy.ndarray): A 3x3 matrix representing the reciprocal lattice vectors.

    Returns:
    numpy.ndarray: The d-spacing values for the given Miller indices.
    """
    Q_hkl = np.linalg.norm(calculate_Q_hkl(hkl, reciprocal_lattice), axis=1)
    dspacing_values = 2*np.pi/Q_hkl
    return dspacing_values

def calculate_two_theta(hkl, reciprocal_lattice, wavelength):
    """
    Calculate the two-theta angles for given Miller indices, reciprocal lattice, and wavelength.

    Parameters:
    hkl (numpy.ndarray): An array of Miller indices (h, k, l).
    reciprocal_lattice (numpy.ndarray): A 3x3 matrix representing the reciprocal lattice vectors.
    wavelength (float): The wavelength of the incident X-ray in meters.

    Returns:
    numpy.ndarray: The two-theta angles in degrees for the given Miller indices.
    """
    wavelength = wavelength * 1e10  # Convert wavelength to Angstroms
    Q_hkl = np.linalg.norm(calculate_Q_hkl(hkl, reciprocal_lattice), axis=1)
    two_thetas = np.rad2deg(2 * np.arcsin(wavelength * Q_hkl / (4 * np.pi)))  # In degrees

    return two_thetas

def check_Bragg_condition(Q_hkls, wavelength, E_bandwidth):
    """
    Check if given Q vectors satisfy the Bragg condition for a given wavelength and energy bandwidth.

    Parameters:
    Q_hkls (numpy.ndarray): An array of Q vectors corresponding to the Miller indices.
    wavelength (float): The wavelength of the incident X-ray in meters.
    E_bandwidth (float): The energy bandwidth of the incident X-ray.

    Returns:
    numpy.ndarray: A boolean array indicating which Q vectors satisfy the Bragg condition.
    """
    wavelength = wavelength * 1e10  # Convert wavelength from meters to Angstroms

    ewald_sphere = Ewald_Sphere(wavelength, E_bandwidth)

    ki = np.array([2*np.pi/wavelength, 0, 0]).reshape(1, -1)
    kf = Q_hkls + ki  # Add ki to each Q_hkl

    kf_hkl_module = np.linalg.norm(kf,axis = 1)

    in_bragg_condition = (kf_hkl_module >= ewald_sphere.Get_Inner_Radius()) & (kf_hkl_module <= ewald_sphere.Get_Outer_Radius())
        
    return in_bragg_condition

def diffraction_direction(Q_hkls, detector, wavelength):
    """
    Calculate the diffraction direction for given Q vectors and a detector setup.

    Parameters:
    Q_hkls (numpy.ndarray): An array of Q vectors corresponding to the Miller indices.
    detector (object): An object representing the detector, with attributes:
                       - beam_center: A tuple (x, y) representing the beam center in pixel coordinates.
                       - pixel_size: A tuple (pixel_size_x, pixel_size_y) representing the size of the detector pixels in meters.
                       - tilting_angle: The tilting angle of the detector in degrees.
                       - sample_detector_distance: The distance from the sample to the detector in meters.
                       - Max_Detectable_Z: A method that returns the maximum detectable Z-coordinate on the detector in meters.
    wavelength (float): The wavelength of the incident X-ray in meters.

    Returns:
    numpy.ndarray: An array of shape (N, 3) containing the diffraction directions (dx, dy, dz) in meters for each Q vector.
    """

    beam_center = (-detector.beam_center[0]*detector.pixel_size[0], detector.Max_Detectable_Z() - detector.beam_center[1]*detector.pixel_size[1]) #This is to make the (0,0) the upper left corner

    
    tilting_angle = np.radians(detector.tilting_angle)

    sample_detector_distance = detector.sample_detector_distance

    wavelength = wavelength*1e10 #Going from m to Å

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
        denominator = kfxz[:,0]
        mask = denominator != 0
        dz[mask] = ((kfxz[:, 2][mask]/denominator)*sample_detector_distance + beam_center[1])/((kfxz[:, 2][mask]/denominator)* np.sin(tilting_angle) + np.cos(tilting_angle))

    except ZeroDivisionError:
        pass

    try:
        denominator = kfxy[:,0]
        mask = denominator != 0

        dy[mask] = (kfxy[:, 1][mask]/denominator)*(sample_detector_distance - dz[mask] * np.sin(tilting_angle)) + beam_center[0]

    except ZeroDivisionError:
        pass

    diffracted_information = np.stack((dx, dy, dz), axis=1)

    return diffracted_information #These in meters

def diffraction_in_detector(diffracted_information, detector):
    """
    Determine if the diffracted directions fall within the detector's detectable area.

    Parameters:
    diffracted_information (numpy.ndarray): An array of shape (N, 3) containing the diffraction directions (dx, dy, dz) in meters.
    detector (object): An object representing the detector, with methods:
                       - Max_Detectable_Y(): Returns the maximum detectable Y-coordinate on the detector in meters.
                       - Min_Detectable_Y(): Returns the minimum detectable Y-coordinate on the detector in meters.
                       - Max_Detectable_Z(): Returns the maximum detectable Z-coordinate on the detector in meters.
                       - Min_Detectable_Z(): Returns the minimum detectable Z-coordinate on the detector in meters.

    Returns:
    numpy.ndarray: A boolean array indicating whether each diffracted direction falls within the detector's detectable area.
    """
    mask = (
        (diffracted_information[:,0] > 0) & 
        (diffracted_information[:,1] <= detector.Max_Detectable_Y()) & 
        (diffracted_information[:,1] >= detector.Min_Detectable_Y()) & 
        (diffracted_information[:,2] <= detector.Max_Detectable_Z()) & 
        (diffracted_information[:,2] >= detector.Min_Detectable_Z())
        )
    return mask

def single_crystal_orientation(phase, wavelength, detector, sample_detector_distance, beam_center,
                               hkls, rotations, y_coordinates, z_coordinates,
                               crystal_orient_guess, tilting_angle=0, rotation_order="xyz", binning=(1,1)):
    """
    Retrieve the single crystal orientation given 3 Bragg reflections.

    Parameters:
    phase (str): The phase of the crystal, either "Hexagonal" or "Monoclinic".
    wavelength (float): The wavelength of the incident X-ray in meters.
    detector (str): The type of the detector.
    sample_detector_distance (float): The distance from the sample to the detector in meters.
    beam_center (tuple): The beam center coordinates in pixel coordinates.
    hkls (list of lists): A list of Miller indices.
    rotations (list of lists): A list of rotations applied to the crystal in degrees.
    y_coordinates (list of floats): A list of y coordinates of the detected spots in pixels.
    z_coordinates (list of floats): A list of z coordinates of the detected spots in pixels.
    crystal_orient_guess (list of floats): An initial guess for the crystal orientation matrix.
    tilting_angle (float, optional): The tilting angle of the detector in degrees. Default is 0.
    rotation_order (str, optional): The order of rotations. Default is "xyz".
    binning (tuple, optional): The binning of the detector pixels. Default is (1, 1).

    Returns:
    numpy.ndarray: A 3x3 matrix representing the crystal orientation.
    """

    detector = Detector(detector_type = detector, sample_detector_distance=sample_detector_distance, tilting_angle=tilting_angle, beam_center = beam_center, binning = binning)

    beam_center = (-detector.beam_center[0]*detector.pixel_size[0], detector.Max_Detectable_Z() - detector.beam_center[1]*detector.pixel_size[1]) #This is to make the (0,0) the upper left corner

    y_distances = np.array(y_coordinates)*(-detector.pixel_size[0]) #in m
    #z_distances = np.array(z_coordinates)*(detector.pixel_size[1])  #in m
    z_distances = detector.Max_Detectable_Z() - np.array(z_coordinates)*(detector.pixel_size[1])  #in m

    wavelength = wavelength*1e10 #Transforming to Å

    def construct_kf_exp(wavelength, y_distance_from_center, z_distance_from_center, sample_detector_distance, tilting_angle = 0):
        
        tilting_angle = np.deg2rad(tilting_angle)

        ki = 2*np.pi/wavelength

        kfy = ki*(y_distance_from_center - beam_center[0])/np.sqrt((z_distance_from_center*np.sin(tilting_angle) - sample_detector_distance)**2 + (beam_center[0] - y_distance_from_center)**2 + (beam_center[1] - z_distance_from_center*np.cos(tilting_angle))**2)

        kfz = np.sqrt(ki**2 - kfy**2)*(beam_center[1] - z_distance_from_center*np.cos(tilting_angle))/np.sqrt((z_distance_from_center*np.sin(tilting_angle) - sample_detector_distance)**2 + (beam_center[1] - z_distance_from_center*np.cos(tilting_angle))**2)


        kfx = np.sqrt(ki**2 - (kfy + kfz))
        return np.array([kfx, kfy, kfz]) #in Å^(-1)
    
    if phase == "Hexagonal":
        lattice = Hexagonal_Lattice()
        bounds = ([-lattice.c] * 9, [lattice.c] * 9)
    
    elif phase == "Monoclinic":
        lattice = Monoclinic_Lattice()
        bounds = ([-lattice.a] * 9, [lattice.a] * 9)

    a, b, c, alpha, beta, gamma = lattice.a, lattice.b, lattice.c, lattice.alpha, lattice.beta, lattice.gamma

    ki = 2*np.pi/wavelength #Transforming wavelength to Å 

    def residuals(params):
        A, B, C, D, E, F, G, H, I = params
        Cryst_Orient = np.array([[A, B, C], [D, E, F], [G, H, I]])
        
        ress = []

        constraint1 = np.linalg.norm(Cryst_Orient[0]) - a
        constraint2 = np.linalg.norm(Cryst_Orient[1]) - b
        constraint3 = np.linalg.norm(Cryst_Orient[2]) - c
        constraint4 = np.dot(Cryst_Orient[0],  Cryst_Orient[1])   - a*b*np.cos(np.deg2rad(gamma))
        constraint5 = np.dot(Cryst_Orient[1],  Cryst_Orient[2])   - b*c*np.cos(np.deg2rad(alpha))
        constraint6 = np.dot(Cryst_Orient[2],  Cryst_Orient[0])   - c*a*np.cos(np.deg2rad(beta ))
        constraints = [constraint1, constraint2, constraint3, constraint4, constraint5, constraint6]

        for i in range(len(hkls)):
            kf = construct_kf_exp(wavelength, y_distances[i], z_distances[i], sample_detector_distance = sample_detector_distance, tilting_angle = tilting_angle)

            Cryst_Orientation = apply_rotation(initial_matrix = Cryst_Orient, rotx = rotations[i][0], roty = rotations[i][1], rotz = rotations[i][2], rotation_order=rotation_order)
            reciprocal_lattice = cal_reciprocal_lattice(Cryst_Orientation)
            GG_peak = calculate_Q_hkl(hkls[i], reciprocal_lattice)

            res_1 =  GG_peak[0] + ki - kf[0]
            res_2 =  GG_peak[1]      - kf[1]
            res_3 =  GG_peak[2]      - kf[2]

            # Restrictions
            ress.append(res_1)
            ress.append(res_2)
            ress.append(res_3)

            # Constraints

            constraint_1 = np.linalg.norm(Cryst_Orientation[0]) - a
            constraint_2 = np.linalg.norm(Cryst_Orientation[1]) - b
            constraint_3 = np.linalg.norm(Cryst_Orientation[2]) - c
            constraint_4 = np.dot(Cryst_Orientation[0],  Cryst_Orientation[1])   - a*b*np.cos(np.deg2rad(gamma))
            constraint_5 = np.dot(Cryst_Orientation[1],  Cryst_Orientation[2])   - b*c*np.cos(np.deg2rad(alpha))
            constraint_6 = np.dot(Cryst_Orientation[2],  Cryst_Orientation[0])   - c*a*np.cos(np.deg2rad(beta ))
            constraints.append(constraint_1)
            constraints.append(constraint_2)
            constraints.append(constraint_3)
            constraints.append(constraint_4)
            constraints.append(constraint_5)
            constraints.append(constraint_6)

        return  np.concatenate((ress, constraints))


    sol = least_squares(residuals, crystal_orient_guess, bounds=bounds, verbose = 2)
    #sol = least_squares(residuals, crystal_orient_guess, verbose = 2, method = "lm")

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
        """
        Initialize a lattice structure with its parameters and initial crystal orientation.

        Parameters:
        a (float): Length of lattice vector a.
        b (float): Length of lattice vector b.
        c (float): Length of lattice vector c.
        alpha (float): Angle between lattice vectors b and c in degrees.
        beta (float): Angle between lattice vectors a and c in degrees.
        gamma (float): Angle between lattice vectors a and b in degrees.
        initial_crystal_orientation (numpy.ndarray): Initial crystal orientation matrix.
        Vanadium_fractional_position (numpy.ndarray): Fractional coordinates of Vanadium atoms.
        Oxygen_fractional_position (numpy.ndarray): Fractional coordinates of Oxygen atoms.
        """
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

    def Apply_Rotation(self, rotx=0, roty=0, rotz=0, rotation_order="xyz"):
        """
        Apply rotation to the crystal orientation matrix.

        Parameters:
        rotx (float): Rotation angle around the x-axis in degrees.
        roty (float): Rotation angle around the y-axis in degrees.
        rotz (float): Rotation angle around the z-axis in degrees.
        rotation_order (str): Order of rotation, e.g., "xyz".

        Returns:
        None
        """
        self.crystal_orientation = np.round(apply_rotation(self.initial_crystal_orientation, rotx, roty, rotz, rotation_order=rotation_order),5)
        self.reciprocal_lattice = cal_reciprocal_lattice(self.crystal_orientation)
    
class Hexagonal_Lattice(Lattice_Structure):
    def __init__(self, a=None, b=None, c=None, alpha=None, beta=None, gamma=None, initial_crystal_orientation=None):
        """
        Initialize a hexagonal lattice structure with default or provided parameters.

        Parameters:
        a (float): Length of lattice vector a. Defaults to 4.9525.
        b (float): Length of lattice vector b. Defaults to the same as a.
        c (float): Length of lattice vector c. Defaults to 14.00093.
        alpha (float): Angle between lattice vectors b and c in degrees. Defaults to 90.
        beta (float): Angle between lattice vectors a and c in degrees. Defaults to 90.
        gamma (float): Angle between lattice vectors a and b in degrees. Defaults to 120.
        initial_crystal_orientation (numpy.ndarray): Initial crystal orientation matrix. Defaults to a predefined value.
        """

        if a is None:
            a = 4.9525
            #a = 4.954
        
        if b is None:
            b = a

        if c is None:
            c = 14.00093
            #c = 14.01
        
        if alpha is None:
            alpha = 90

        if beta is None:
            beta = 90
        
        if gamma is None:
            gamma = 120

        #V = np.sin(np.deg2rad(60))*a*b*c
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
    def __init__(self, a=None, b=None, c=None, alpha=None, beta=None, gamma=None, initial_crystal_orientation=None):
        """
        Initialize a monoclinic lattice structure with default or provided parameters.

        Parameters:
        a (float): Length of lattice vector a. Defaults to 7.266.
        b (float): Length of lattice vector b. Defaults to 5.0024.
        c (float): Length of lattice vector c. Defaults to 5.5479.
        alpha (float): Angle between lattice vectors b and c in degrees. Defaults to 90.
        beta (float): Angle between lattice vectors a and c in degrees. Defaults to 96.760.
        gamma (float): Angle between lattice vectors a and b in degrees. Defaults to 90.
        initial_crystal_orientation (numpy.ndarray): Initial crystal orientation matrix. Defaults to a predefined value.
        """
        
        if a is None:
            a = 7.266
        
        if b is None:
            b = 5.0024

        if c is None:
            c = 5.5479
        
        if alpha is None:
            alpha = 90

        if beta is None:
            beta = 96.760
        
        if gamma is None:
            gamma = 90
        
        #V_M = a*b*c*np.sin(np.deg2rad(beta))

        if initial_crystal_orientation is None:
            #initial_crystal_orientation = np.array([
            #[b/2 - np.sqrt((c**2 + a*c*np.cos(np.radians(beta)) - (b**2)/4)/3), 
            #b/np.sqrt(12) - np.sqrt(c**2 + a*c*np.cos(np.radians(beta)) - (b**2)/4),  
            #np.sqrt((14*c**2 - 4*a**2 + a*c*np.cos(np.radians(beta)))/9)],
            #[b,0,0],
            #[b/4 - np.sqrt((c**2 + a*c*np.cos(np.radians(beta)) - (b**2)/4)/12),
            #b/np.sqrt(48) - 0.5*np.sqrt(c**2 + a*c*np.cos(np.radians(beta)) - (b**2)/4), 
            #-np.sqrt((14*c**2 - 4*a**2 + a*c*np.cos(np.radians(beta)))/9)]
            #])

            initial_crystal_orientation = np.array([
            [0, (2/np.sqrt(3))*np.sqrt(a*c*np.cos(np.radians(beta)) + c**2), (1/np.sqrt(3))*np.sqrt(a**2 - 2*a*c*np.cos(np.radians(beta))) ],
            [a,                           0                                ,                                0                              ],
            [0, (1/np.sqrt(3))*np.sqrt(a*c*np.cos(np.radians(beta)) + c**2), -(1/np.sqrt(3))*np.sqrt(a**2 - 2*a*c*np.cos(np.radians(beta)))]
            ])

        Vanadium_fractional_position = np.array([0.34460,    0.00500,    0.29850])
        Oxygen_fractional_position   = np.array([[0.41100,    0.84700,    0.65000], [0.25000,    0.18000,    0.00000]])

        super().__init__(a, b, c, alpha, beta, gamma, initial_crystal_orientation, Vanadium_fractional_position, Oxygen_fractional_position)

class Detector:
    def __init__(self, detector_type, sample_detector_distance=None, tilting_angle=0, beam_center=(0, 0), margin=0, binning=(1, 1)):
        """
        Initialize a Detector instance.

        Parameters:
        detector_type (str): Type of the detector, e.g., "Rayonix", "Pilatus".
        sample_detector_distance (float): Sample-detector distance in meters.
        tilting_angle (float): Angle of tilting of the detector in degrees.
        beam_center (tuple): Coordinates of the beam center on the detector in pixels.
        margin (float): Margin percentage for adjusting detectable region.
        pixel_size (tuple): Size of pixels in meters.
        binning (tuple): Binning factors for the detector.

        Returns:
        None
        """
        # Store parameters
        self.detector_type = detector_type #I just want to make it accessible 
        self.beam_center = beam_center #In pixel number
        self.margin = margin
        self.binning = binning
        self.tilting_angle = tilting_angle

        # Convert tilting angle to radians
        tilting_angle = np.deg2rad(tilting_angle)
        
        try:
            # Use pyFAI to get detector dimensions and pixel sizes
            det = pyFAI.detectors.detector_factory("RayonixmX170")
            self.height = (det.MAX_SHAPE[0])*(det.pixel1)
            self.width = (det.MAX_SHAPE[1])*(det.pixel2)
            self.pixel_size = (det.pixel1*binning[0], det.pixel2*binning[1])

        except NameError:

            # Set dimensions and pixel size based on detector type
            if self.detector_type == "Rayonix":
                self.height = 0.170 #m
                self.width = 0.170 #m
                self.pixel_size = (44e-6*binning[0], 44e-6*binning[1])

            elif self.detector_type == "Pilatus":
                self.height = 0.142 #m
                self.width = 0.253 #m
                self.pixel_size = (172e-6*binning[0], 172e-6*binning[1])
            
            else:
                print("Invalid Detector")
            
            #elif self.detector_type == "Rigaku":
            #    self.height = 80
            #    self.width  = 80
            
        
        if self.detector_type == "Pilatus":
            self.height = 0.142 #m
            self.width = 0.253 #m
            self.pixel_size = (172e-6*binning[0], 172e-6*binning[1])
        
        # Set sample-detector distance if provided
        if sample_detector_distance is not None:
            self.sample_detector_distance = sample_detector_distance
        
        
    def Max_Detectable_Z(self):
        """
        Compute the maximum detectable Z-coordinate on the detector.

        Returns:
        float: Maximum Z-coordinate.
        """
        max_dz = (1 - self.margin/100)*(self.height)#*np.sin(np.pi/2 - self.tilting_angle)

        return max_dz
    
    def Min_Detectable_Z(self):
        """
        Compute the minimum detectable Z-coordinate on the detector.

        Returns:
        float: Minimum Z-coordinate.
        """
        min_dz = (self.height)*(self.margin/100)

        return min_dz
    
    def Max_Detectable_Y(self):
        """
        Compute the maximum detectable Y-coordinate on the detector.

        Returns:
        float: Maximum Y-coordinate.
        """
        max_dy = -(self.width)*(self.margin/100)
        return max_dy

    def Min_Detectable_Y(self):
        """
        Compute the minimum detectable Y-coordinate on the detector.

        Returns:
        float: Minimum Y-coordinate.
        """
        min_dy = -(self.width)*(1 - self.margin/100)
        return min_dy
        
class Ewald_Sphere:
    def __init__(self, wavelength, E_bandwidth):
        """
        Initialize an Ewald Sphere instance.

        Parameters:
        wavelength (float): Wavelength of the incident radiation.
        E_bandwidth (float): Energy bandwidth in percentage.

        Returns:
        None
        """
        self.wavelength = wavelength
        self.E_bandwidth = E_bandwidth
        self.radius = 2*np.pi/wavelength
        self.radius_inner = self.radius*(1 - (E_bandwidth/2)/100)
        self.radius_outer = self.radius*(1 + (E_bandwidth/2)/100)

    def Get_Radius(self):
        """
        Get the radius of the Ewald sphere.

        Returns:
        float: Radius of the Ewald sphere.
        """
        return self.radius

    def Get_Inner_Radius(self):
        """
        Get the inner radius of the Ewald sphere.

        Returns:
        float: Inner radius of the Ewald sphere.
        """
        return self.radius_inner
        
    def Get_Outer_Radius(self):
        """
        Get the outer radius of the Ewald sphere.

        Returns:
        float: Outer radius of the Ewald sphere.
        """
        return self.radius_outer

    def Generate_Ewald_Sphere_Data(self):
        """
        Generate data points for visualizing the Ewald sphere.

        Returns:
        tuple: Arrays containing x, y, and z coordinates of points on the Ewald sphere.
        """
        theta = np.linspace(0, 2 * np.pi, 100)
        phi = np.linspace(0, np.pi, 100)
        theta, phi = np.meshgrid(theta, phi)

        x = self.radius*np.sin(phi)*np.cos(theta)
        y = self.radius*np.sin(phi)*np.sin(theta)
        z = self.radius*np.cos(phi)

        return x, y, z

    def Add_To_Existing_Plot(self, existing_fig):
        """
        Add the Ewald sphere to an existing plot.

        Parameters:
        existing_fig (plotly.graph_objs.Figure): Existing plotly figure.

        Returns:
        plotly.graph_objs.Figure: Figure with the Ewald sphere added.
        """
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
        """
        Initialize an Atom instance.

        Parameters:
        symbol (str): Symbol of the atom.
        atomic_structure_factor (numpy.ndarray): Array containing pairs of distances and corresponding atomic structure factors.

        Returns:
        None
        """
        self.Symbol = symbol
        self.Atomic_Structure_Factor = atomic_structure_factor

class Vanadium(Atom):
    def __init__(self):
        """
        Initialize a Vanadium instance.

        Parameters:
        None

        Returns:
        None
        """
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
        """
        Initialize an Oxygen instance.

        Parameters:
        None

        Returns:
        None
        """
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



def atomic_structure_factor(atomic_structure_factor_list, wavelength, two_theta):
    """
    Calculate the atomic structure factor for a given wavelength and scattering angle.

    Parameters:
    atomic_structure_factor_list (numpy.ndarray): A 2D array where the first column represents sin(theta)/lambda values 
                                                  and the second column represents the corresponding atomic structure factors.
    wavelength (float): The wavelength of the incident X-ray in meters.
    two_theta (float): The scattering angle in degrees.

    Returns:
    float: The atomic structure factor corresponding to the given wavelength and two_theta.
    """
    sin_theta_over_lambda = np.sin((np.pi/180)*two_theta/2)/wavelength
    dummy = np.argmin(np.abs(atomic_structure_factor_list[:,0] - sin_theta_over_lambda))
    atomic_structure_factor = atomic_structure_factor_list[dummy][1]
    return atomic_structure_factor

def structure_factor_given_atom(atomic_factor, hkl, atomic_positions):
    """
    Calculate the structure factor for a given set of atomic positions and an atomic scattering factor.

    Parameters:
    atomic_factor (float): The atomic scattering factor for a particular atom.
    hkl (list or numpy.ndarray): The Miller indices [h, k, l].
    atomic_positions (list of lists or numpy.ndarray): A list of atomic positions in fractional coordinates.

    Returns:
    float: The structure factor for the given atomic positions and atomic scattering factor.
    """
    dummy = []
    for atomic_position in atomic_positions:
        bla = atomic_factor*np.exp(2j*np.pi*np.dot(hkl, atomic_position))
        dummy.append(bla*np.conj(bla))
    structure_factor_atom = sum(dummy)

    return structure_factor_atom

def calculate_angle_between_two_reflections(phase, hkl_1, hkl_2):
    """
    Calculate the angle between two reflections for a given crystal phase.

    Parameters:
    phase (str): The phase of the crystal, either "Hexagonal" or "Monoclinic".
    hkl_1 (list or numpy.ndarray): The Miller indices [h, k, l] of the first reflection.
    hkl_2 (list or numpy.ndarray): The Miller indices [h, k, l] of the second reflection.

    Returns:
    float: The angle between the two reflections in degrees.
    """

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

"""
def single_crystal_orientation(phase, wavelength, detector, sample_detector_distance, beam_center,
                               hkls, rotations, y_coordinates, z_coordinates,
                               crystal_orient_guess, tilting_angle = 0, rotation_order = "xyz", binning = (1,1)):

    #This long function is to be used when trying to retreive single crystal orientation given 3 Braggs.

    detector = Detector(detector_type = detector, sample_detector_distance=sample_detector_distance, tilting_angle=tilting_angle, beam_center = beam_center, binning = binning)

    beam_center = (beam_center[0]*(-detector.pixel_size[0]),beam_center[1]*(detector.pixel_size[1])) #in m

    y_distances = np.array(y_coordinates)*(-detector.pixel_size[0]) #in m
    z_distances = np.array(z_coordinates)*(detector.pixel_size[1])  #in m

    wavelength = wavelength*1e10 #Transforming to Å

    def construct_kf_exp(wavelength, y_distance_from_center, z_distance_from_center, sample_detector_distance, tilting_angle = 0):
        
        tilting_angle = np.deg2rad(tilting_angle)

        ki = 2*np.pi/wavelength

        kfy_sq = (ki**2)*(((beam_center[0] - y_distance_from_center)**2)/((z_distance_from_center*np.sin(tilting_angle) - sample_detector_distance)**2 + (beam_center[0] - y_distance_from_center)**2 + (beam_center[1] - z_distance_from_center*np.cos(tilting_angle))**2))

        kfz_sq = (ki**2 - kfy_sq**2)*(((beam_center[1] - z_distance_from_center*np.cos(tilting_angle))**2)/((z_distance_from_center*np.sin(tilting_angle) - sample_detector_distance)**2 + (beam_center[1] - z_distance_from_center*np.cos(tilting_angle))**2))


        kfx_sq = ki**2 - (kfy_sq + kfz_sq)
        return np.array([kfx_sq, kfy_sq, kfz_sq]) #in Å^(-1)
    
    if phase == "Hexagonal":
        lattice = Hexagonal_Lattice()
        bounds = ([-lattice.c] * 9, [lattice.c] * 9)
    
    elif phase == "Monoclinic":
        lattice = Monoclinic_Lattice()
        bounds = ([-lattice.a] * 9, [lattice.a] * 9)

    a, b, c, alpha, beta, gamma = lattice.a, lattice.b, lattice.c, lattice.alpha, lattice.beta, lattice.gamma

    ki = 2*np.pi/wavelength #Transforming wavelength to Å 

    def residuals(params):
        A, B, C, D, E, F, G, H, I = params
        Cryst_Orient = np.array([[A, B, C], [D, E, F], [G, H, I]])
        
        ress = []

        constraint1 = np.linalg.norm(Cryst_Orient[0]) - a
        constraint2 = np.linalg.norm(Cryst_Orient[1]) - b
        constraint3 = np.linalg.norm(Cryst_Orient[2]) - c
        constraint4 = np.dot(Cryst_Orient[0],  Cryst_Orient[1])   - a*b*np.cos(np.deg2rad(gamma))
        constraint5 = np.dot(Cryst_Orient[1],  Cryst_Orient[2])   - b*c*np.cos(np.deg2rad(alpha))
        constraint6 = np.dot(Cryst_Orient[2],  Cryst_Orient[0])   - c*a*np.cos(np.deg2rad(beta ))
        constraints = [constraint1, constraint2, constraint3, constraint4, constraint5, constraint6]

        for i in range(len(hkls)):
            kf_sq = construct_kf_exp(wavelength, y_distances[i], z_distances[i], sample_detector_distance = sample_detector_distance, tilting_angle = tilting_angle)

            Cryst_Orientation = apply_rotation(initial_matrix = Cryst_Orient, rotx = rotations[i][0], roty = rotations[i][1], rotz = rotations[i][2], rotation_order=rotation_order)
            reciprocal_lattice = cal_reciprocal_lattice(Cryst_Orientation)
            GG_peak = calculate_Q_hkl(hkls[i], reciprocal_lattice)

            res_1 = (GG_peak[0] + ki)**2 - kf_sq[0]
            res_2 =  GG_peak[1]**2       - kf_sq[1]
            res_3 =  GG_peak[2]**2       - kf_sq[2]

            # Restrictions
            ress.append(res_1)
            ress.append(res_2)
            ress.append(res_3)

            # Constraints

            constraint_1 = np.linalg.norm(Cryst_Orientation[0]) - a
            constraint_2 = np.linalg.norm(Cryst_Orientation[1]) - b
            constraint_3 = np.linalg.norm(Cryst_Orientation[2]) - c
            constraint_4 = np.dot(Cryst_Orientation[0],  Cryst_Orientation[1])   - a*b*np.cos(np.deg2rad(gamma))
            constraint_5 = np.dot(Cryst_Orientation[1],  Cryst_Orientation[2])   - b*c*np.cos(np.deg2rad(alpha))
            constraint_6 = np.dot(Cryst_Orientation[2],  Cryst_Orientation[0])   - c*a*np.cos(np.deg2rad(beta ))
            constraints.append(constraint_1)
            constraints.append(constraint_2)
            constraints.append(constraint_3)
            constraints.append(constraint_4)
            constraints.append(constraint_5)
            constraints.append(constraint_6)

        return  np.concatenate((ress, constraints))


    sol = least_squares(residuals, crystal_orient_guess, bounds=bounds, verbose = 2)
    #sol = least_squares(residuals, crystal_orient_guess, verbose = 2, method = "lm")

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
"""

"""
def diffraction_direction(Q_hkls, detector, wavelength):
    #sample_detector_distance is in meters. So dx, dy, dz will be too.
    
    beam_center = (-detector.beam_center[0]*detector.pixel_size[0], detector.beam_center[1]*detector.pixel_size[1])

    tilting_angle = np.radians(detector.tilting_angle)

    sample_detector_distance = detector.sample_detector_distance
    #print(sample_detector_distance)
    #print(beam_center)

    wavelength = wavelength*1e10 #Going from m to Å

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
        denominator = kfxz[:,0]
        mask = denominator != 0
        dz[mask] = ((kfxz[:, 2][mask]/denominator)*sample_detector_distance + beam_center[1])/((kfxz[:, 2][mask]/denominator)* np.sin(tilting_angle) + np.cos(tilting_angle))

    except ZeroDivisionError:
        pass

    try:
        denominator = kfxy[:,0]
        mask = denominator != 0

        dy[mask] = (kfxy[:, 1][mask]/denominator)*(sample_detector_distance - dz[mask] * np.sin(tilting_angle)) + beam_center[0]

    except ZeroDivisionError:
        pass

    diffracted_information = np.stack((dx, dy, dz), axis=1)

    return diffracted_information
"""