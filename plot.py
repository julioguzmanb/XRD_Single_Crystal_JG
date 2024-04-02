import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import XRD_Single_Crystal_JG.utils as utils


def Colorize(vector, vmin=None, vmax=None, ax=None, cmap=plt.cm.jet):
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

def plot_reciprocal(Q_hkls, hkls, wavelength, E_bandwidth):

    ewald_sphere = utils.Ewald_Sphere(wavelength, E_bandwidth)
    ki = np.array([ewald_sphere.Get_Radius(), 0, 0])

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

    for w, Q_hkl in enumerate(Q_hkls):
        x, y, z = Q_hkl + ki

        if utils.check_Bragg_condition(Q_hkl, wavelength, E_bandwidth) == True: 
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
            "text": "Reciprocal Lattice",
            "x": 0.5,
            "y": 0.9,  
            "xanchor": 'center', 
            "yanchor": 'top', 
            "font": {'size': 38} 
        }
    )

    fig.show()

def plot_detector(data, colorize = False):
    data["detector"]
    fig_size = (8*(data["detector"].height/170)*data["detector"].width/data["detector"].height + 0.3, 8*(data["detector"].height/170))
    plt.figure(figsize = (fig_size[0], fig_size[1]))
    plt.rcParams.update({'font.size': 16})

    a_x, a_y, a_z = np.round(data["crystal"]["lattice_params"][0], 3)
    b_x, b_y, b_z = np.round(data["crystal"]["lattice_params"][1], 3)
    c_x, c_y, c_z = np.round(data["crystal"]["lattice_params"][2], 3)

    plt.title("Detector = %s, Phase = %s, $\\phi$ = %s°\na = (%s x, %s y, %s z)\nb = (%s x, %s y, %s z)\nc = (%s x, %s y, %s z)\n rotations: %s°$\parallel$ x, %s°$\parallel$ y, %s °$\parallel$ z"%(data["detector"].detector_type, data["crystal"]["phase"], np.round(np.rad2deg(data["detector"].tilting_angle),1), a_x, a_y, a_z, b_x, b_y, b_z, c_x, c_y, c_z, data["crystal"]["orientation"][0], data["crystal"]["orientation"][1], data["crystal"]["orientation"][2]))
    
    [plt.scatter(y_val, z_val, label=label) for y_val, z_val, label in zip(data["y_coordinate"], data["z_coordinate"], data["hkls"])]

    if colorize == True:
        Colorize(vector = list(range(len(data["hkls"]))), cmap = plt.cm.jet)
    
    plt.legend(title = "(h,k,l)", loc = "upper right", fontsize = 14, framealpha = 1)
    plt.xlim(-0.5*data["detector"].width, 0.5*data["detector"].width)
    plt.xlabel("Detector Width [mm]",fontsize = 16)
    plt.ylim(0, data["detector"].height)
    plt.ylabel("Detector Height [mm]",fontsize = 16)
    plt.tight_layout()
    plt.grid()
    plt.show()


