import numpy as np
#import plotly.graph_objs as go
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import XRD_Single_Crystal_JG.utils as utils


def Colorize(vector, vmin=None, vmax=None, ax=None, cmap=plt.cm.jet):
    """
    Colorize plotted data based on a vector of values.

    Parameters:
    vector (numpy.ndarray): Array of values to be colorized.
    vmin (float, optional): Minimum value for normalization. Defaults to None.
    vmax (float, optional): Maximum value for normalization. Defaults to None.
    ax (matplotlib.axes.Axes, optional): Axes object to apply colorization. Defaults to None.
    cmap (matplotlib.colors.Colormap, optional): Colormap to use for colorization. Defaults to plt.cm.jet.

    Returns:
    None
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


def plot_reciprocal(Q_hkls, hkls, wavelength, E_bandwidth):
    """
    Plot the reciprocal space and Ewald construction.

    Parameters:
    - Q_hkls (numpy.ndarray): Array of shape (N, 3) containing the scattering vectors.
    - hkls (numpy.ndarray): Array of shape (N, 3) containing the Miller indices for each reflection.
    - wavelength (float): Wavelength of the incident X-ray beam in meters.
    - E_bandwidth (float): Energy bandwidth of the X-ray beam in percentage.

    Returns:
    - None
      Displays the 3D plot.
    """
    wavelength = wavelength*1e10

    ewald_sphere = utils.Ewald_Sphere(wavelength, E_bandwidth)
    ki = np.array([2*np.pi/wavelength, 0, 0]).reshape(1, -1)

    kf_hkls = Q_hkls + ki

    in_bragg_condition = utils.check_Bragg_condition(Q_hkls, wavelength*1e-10, E_bandwidth)

    fig = plt.figure(figsize=(10, 8))
    plt.rcParams.update({'font.size': 15})

    ax = fig.add_subplot(111, projection='3d')

    for i, (x,y,z) in enumerate(kf_hkls[in_bragg_condition]):
        ax.scatter(x, y, z, label="(%s)"%(str(hkls.tolist()[i]).replace("[","").replace("]","").replace(",","")), s = 40)  # Plot points with colors

    if len(hkls[in_bragg_condition]) > 1:
        Colorize(vector = list(range(len(hkls[in_bragg_condition]))),cmap=plt.cm.jet, ax = ax)

    ax.scatter(0, 0, 0, c='black', label='Ewald Center', s = 100)  # Plot center of Ewald sphere

    #ax.legend(fontsize = 13, framealpha = 1, title = "(hkl) in Bragg C.")
    legend = ax.legend(fontsize = 13, framealpha = 1, title = "(hkl) in Bragg c.")
    title = legend.get_title()
    title.set_fontsize(14)

    #Plotting data that is not in Bragg condition
    x, y, z = kf_hkls[~in_bragg_condition].T  # Transpose to get x, y, z separately
    ax.scatter(x, y, z, c = "blue", s = 20)  # Plot points with colors

    # Plot Ewald sphere
    u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:50j]
    x_ewald = ewald_sphere.Get_Radius() * np.cos(u) * np.sin(v)
    y_ewald = ewald_sphere.Get_Radius() * np.sin(u) * np.sin(v)
    z_ewald = ewald_sphere.Get_Radius() * np.cos(v)
    ax.plot_surface(x_ewald, y_ewald, z_ewald, color='lightgreen', alpha=0.15, linewidth=0)

    # Add labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Add title
    ax.set_title('Ewald Construction')

    # Set aspect ratio
    ax.set_box_aspect([1,1,1])

    # Set axis limits
    lim = int(np.linalg.norm(ki)*1.2)
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])

    # Adjust view angle manually
    ax.view_init(elev=30, azim=45)

    # Initialize mouse controls for zooming
    ax.mouse_init()

    plt.tight_layout()

    plt.show()


def plot_detector(data, colorize = False):
    """
    Plot the detector layout.

    Parameters:
    - data (dict): Dictionary containing detector information.
                   Requires keys: "detector", "y_coordinate", "z_coordinate", "hkls".
    - colorize (bool): If True, colorizes the points on the detector plot based on the Miller indices.
                       Default is False.

    Returns:
    - None
      Displays the detector plot.
    """
    detector = data["detector"]
    
    if colorize == True:
        [plt.scatter(y_val, z_val, label=label) for y_val, z_val, label in zip(data["y_coordinate"], data["z_coordinate"], data["hkls"])]
        Colorize(vector = list(range(len(data["hkls"]))), cmap = plt.cm.jet)

    else:
        [plt.scatter(y_val, z_val, label=label, color = "blue") for y_val, z_val, label in zip(data["y_coordinate"], data["z_coordinate"], data["hkls"])]
    
    plt.scatter(detector.beam_center[0], detector.beam_center[1], label = "Beam Center",marker='x', color='black', s = 100)
    plt.legend(title = "(h,k,l)", loc = "upper right", fontsize = 14, framealpha = 0.4)

    plt.xlim(abs(detector.Max_Detectable_Y()/detector.pixel_size[0]), abs(detector.Min_Detectable_Y()/detector.pixel_size[0]))
    plt.xlabel("y-direction [pixel]",fontsize = 16)
    plt.ylim(detector.Min_Detectable_Z()/detector.pixel_size[1], detector.Max_Detectable_Z()/detector.pixel_size[1])
    plt.ylabel("z-direction [pixel]",fontsize = 16)
    plt.tight_layout()
    plt.grid()
    plt.show()
    
def plot_guidelines(hkls, lattice_structure, detector, wavelength):
    """
    Plot diffraction circle guidelines on the detector.

    Parameters:
    - hkls (numpy.ndarray): Array of Miller indices (hkl) for the diffraction circles.
    - lattice_structure (Lattice_Structure): Object representing the crystal lattice structure.
    - detector (Detector): Object representing the detector.
    - wavelength (float): Wavelength of incident X-ray radiation.

    Returns:
    - None
      Displays the diffraction circle guidelines on the detector plot.
    """
    # Calculate two theta angles
    two_theta = utils.calculate_two_theta(hkl = hkls, reciprocal_lattice=lattice_structure.reciprocal_lattice, wavelength=wavelength)

    # Calculate distances from sample to detector
    r = detector.sample_detector_distance*np.tan(np.radians(two_theta))

    # Generate angles
    theta = np.linspace(0, 2*np.pi, 100)

    # Calculate y and z coordinates for a circle
    y = r * np.cos(theta)/(-detector.pixel_size[0])
    z = r * np.sin(theta)/(detector.pixel_size[1])

    # Function to distort circle based on detector parameters
    def distort_circle(y,z, detector):
        tilting_angle = np.radians(detector.tilting_angle)

        # Convert y and z coordinates to meters
        y = y*(-detector.pixel_size[0])
        z = z*(detector.pixel_size[1])

        # Calculate beam center in meters
        beam_center = (-detector.beam_center[0]*detector.pixel_size[0], detector.beam_center[1]*detector.pixel_size[1]) #In meters

        # Apply distortion to y and z coordinates
        Z = (z + beam_center[1])/(z*np.sin(tilting_angle)/detector.sample_detector_distance + np.cos(tilting_angle))
        Y = ((detector.sample_detector_distance - beam_center[1]*np.tan(tilting_angle))/(detector.sample_detector_distance + z*np.tan(tilting_angle)))*y + beam_center[0]
        return Y,Z
    
    # Apply distortion to circle coordinates
    Y,Z = distort_circle(y,z, detector)

    # Scale back y and z coordinates
    Y = Y/(-detector.pixel_size[0])
    Z = Z/(detector.pixel_size[1])

    # Plot distorted circle as guidelines
    plt.plot(Y, Z, "--",color = "black", linewidth = 2)


def plot_guidelines(hkls, lattice_structure, detector, wavelength):

    hkls = np.array(hkls)

    two_theta = utils.calculate_two_theta(hkl=hkls, reciprocal_lattice=lattice_structure.reciprocal_lattice, wavelength=wavelength)
    r = detector.sample_detector_distance * np.tan(np.radians(two_theta))

    theta = np.linspace(0, 2 * np.pi, 100)

    # Calculate y and z for each hkl using broadcasting
    y = np.outer(r, np.cos(theta)) / (-detector.pixel_size[0])
    z = np.outer(r, np.sin(theta)) / detector.pixel_size[1]

    def distort_circle(y, z, detector):
        tilting_angle = np.radians(detector.tilting_angle)
        pixel_size = detector.pixel_size

        beam_center = (-detector.beam_center[0]*detector.pixel_size[0], detector.Max_Detectable_Z() - detector.beam_center[1]*detector.pixel_size[1]) #This is to make the (0,0) the upper left corner

        # Convert pixel positions to meters
        y_m = y * (-pixel_size[0])
        z_m = z * pixel_size[1]

        # Compute Z and Y using vectorized operations
        Z = (z_m + beam_center[1]) / (z_m * np.sin(tilting_angle) / detector.sample_detector_distance + np.cos(tilting_angle))
        Y = ((detector.sample_detector_distance - beam_center[1] * np.tan(tilting_angle)) / (detector.sample_detector_distance + z_m * np.tan(tilting_angle))) * y_m + beam_center[0]

        return Y, Z

    # Apply distortion to all circles
    Y, Z = distort_circle(y, z, detector)

    # Convert Y and Z back to pixels
    Y = Y / (-detector.pixel_size[0])
    Z = (detector.Max_Detectable_Z() - Z) / detector.pixel_size[1] #This is to make the (0,0) the upper left corner

    # Plot all distorted circles
    if len(hkls) == 1:
        plt.plot(Y[i], Z[i], "--", color="black", linewidth=2, label = str(hkls[i]).replace("[", "(").replace("]", ")"))

    else:
        hkls_color = np.linspace(0, 1, len(hkls))
        for i in range(len(hkls)):
            plt.plot(Y[i], Z[i], "--", color = plt.cm.jet(hkls_color[i]),linewidth=1, label = str(hkls[i]).replace("[", "(").replace("]", ")"))
    
    plt.legend(title = "(h,k,l)", loc = "upper right", fontsize = 12, framealpha = 0.95)

    plt.show()



"""
def plot_reciprocal(Q_hkls, hkls, wavelength, E_bandwidth):

    ewald_sphere = utils.Ewald_Sphere(wavelength, E_bandwidth)
    #ki = np.array([ewald_sphere.Get_Radius(), 0, 0])
    ki = np.array([2*np.pi/wavelength, 0, 0]).reshape(1, -1)

    kf_hkls = Q_hkls + ki

    in_bragg_condition = utils.check_Bragg_condition(Q_hkls, wavelength, E_bandwidth)

    colors = np.where(in_bragg_condition, 'red', 'blue')

    fig = go.Figure()
    
    fig.add_trace(go.Scatter3d(
        x=[0],
        y=[0],
        z=[0],
        mode = "markers",
        marker = dict(
            size = 8,
            color = "yellow", 
        ),
    ))

    for w, kf_hkl in enumerate(kf_hkls):
        
        x, y, z = kf_hkl

        if utils.check_Bragg_condition(Q_hkls[w], wavelength, E_bandwidth) == True: 
            color = "red"
        else:
            color = "blue"

        fig.add_trace(go.Scatter3d(
            x=[x],
            y=[y],
            z=[z],
            mode = "markers",
            text = [hkls[w]],
            marker = dict(
                size = 8,
                color = color, 
            ),
            name=f'({hkls[w][0]}, {hkls[w][1]}, {hkls[w][2]})' 
        ))

    fig = ewald_sphere.Add_To_Existing_Plot(fig)

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-12, 12]), 
            yaxis=dict(range=[-12, 12]), 
            zaxis=dict(range=[-12, 12]), 
            aspectratio=dict(x=1, y=1, z=1),
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z"
        ),
        title={
            "text": "Ewald Construction",
            "x": 0.5,
            "y": 0.9,  
            "xanchor": 'center', 
            "yanchor": 'top', 
            "font": {'size': 38} 
        }
    )

    fig.show()
"""


