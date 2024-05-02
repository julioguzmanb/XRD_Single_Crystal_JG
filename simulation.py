import numpy as np
import matplotlib.pyplot as plt
import itertools
import XRD_Single_Crystal_JG.utils as utils
import XRD_Single_Crystal_JG.plot as plot
plt.ion()

def reciprocal_space(phase, rotx, roty, rotz, wavelength, E_bandwidth, smallest_number, largest_number, initial_crystal_orientation = None, rotation_order = "xyz"):

    #Creating lattice structure
    if phase == "Monoclinic":
        lattice_structure = utils.Monoclinic_Lattice(initial_crystal_orientation = initial_crystal_orientation)
    
    elif phase == "Hexagonal":
        lattice_structure = utils.Hexagonal_Lattice(initial_crystal_orientation = initial_crystal_orientation) 

    #Rotating the crystal
    lattice_structure.Apply_Rotation(rotx, roty, rotz, rotation_order = rotation_order) #This Apply_Rotation is a method of the object.

    hkls = utils.create_possible_reflections(phase, smallest_number, largest_number)

    #Calculating the momentum transfer vectors
    Q_hkls = utils.calculate_Q_hkl(hkls, lattice_structure.reciprocal_lattice)

    plot.plot_reciprocal(Q_hkls, hkls, wavelength, E_bandwidth)

def detector(phase, rotx, roty, rotz, detector, sample_detector_distance, wavelength, E_bandwidth, smallest_number, largest_number, tilting_angle = 0, initial_crystal_orientation = None, margin = 0, beam_center = (0,0), rotation_order = "xyz"):


    #Creating detector instance
    detector = utils.Detector(detector_type=detector, sample_detector_distance=sample_detector_distance, tilting_angle=tilting_angle, margin = margin, beam_center = beam_center)

    #Creating lattice structure
    if phase == "Monoclinic":
        lattice_structure = utils.Monoclinic_Lattice(initial_crystal_orientation = initial_crystal_orientation)
    
    elif phase == "Hexagonal":
        lattice_structure = utils.Hexagonal_Lattice(initial_crystal_orientation = initial_crystal_orientation)

    #Rotating the crystal
    lattice_structure.Apply_Rotation(rotx, roty, rotz, rotation_order = rotation_order)

    #Calculating the reciprocal lattice vectors
    #a,b,c = lattice_structure.crystal_orientation
    #a_rec, b_rec, c_rec = utils.cal_reciprocal_lattice(a, b, c)

    #Creating all possible hkl reflections
    hkls = utils.create_possible_reflections(phase, smallest_number, largest_number)

    #Calculating the momentum transfer vectors
    Q_hkls = utils.calculate_Q_hkl(hkls, lattice_structure.reciprocal_lattice)
    
    counter = 0

    #Creating the dictionnary for storing the data
    data = {}
    data["crystal"] = {}
    data["crystal"]["phase"] = phase
    data["crystal"]["orientation"] = [rotx, roty, rotz]
    data["crystal"]["lattice_params"] = [lattice_structure.crystal_orientation[0], lattice_structure.crystal_orientation[1], lattice_structure.crystal_orientation[2]]
    data["detector"] = detector
    data["hkls"] = []
    data["y_coordinate"] = []
    data["z_coordinate"] = []

    #Loop for the created Q vectors
    for w, Q_hkl in enumerate(Q_hkls):
        if utils.check_Bragg_condition(Q_hkl, wavelength, E_bandwidth) == True: 
            dx, dy, dz = utils.diffraction_direction(Q_hkl, wavelength, sample_detector_distance, tilting_angle)

            if utils.diffraction_in_detector(dx, dy, dz, detector) == True:
                dy = dy + detector.beam_center[0]
                dz = dz + detector.beam_center[1]
                data["hkls"].append((hkls[w][0], hkls[w][1],hkls[w][2]))   
                data["y_coordinate"].append(dy)
                data["z_coordinate"].append(dz)
                counter += 1
            else:
                pass
        else:
            pass

    if counter == 0:
        print("No (hkl) reflections seen in the detector!!")
    
    elif counter == 1:
        plot.plot_detector(data, beam_center = beam_center)
    
    elif counter >=2:
        plot.plot_detector(data, beam_center = beam_center, colorize = True)

def mapping(phase, detector, sample_detector_distance, wavelength, rot_x_start, rot_x_end, step_rot_x, rot_z_start, rot_z_end, step_rot_z, E_bandwidth, smallest_number = -6, largest_number = 6, tilting_angle = 0, plot_singles = False, plot_doubles = False, plot_triples = False, plot_fourths = False, plot_more_than_four = False, initial_crystal_orientation = None, margin = 0, beam_center = (0,0), rotation_order = "xyz"):

    detector = utils.Detector(detector_type=detector, sample_detector_distance=sample_detector_distance, tilting_angle=tilting_angle, margin = margin, beam_center = beam_center)

    step_rots_x = int(abs((rot_x_start - rot_x_end)/step_rot_x)) +1
    rots_x = np.linspace(rot_x_start, rot_x_end, step_rots_x)
    step_rots_z = int(abs((rot_z_start - rot_z_end)/step_rot_z)) +1
    rots_z = np.linspace(rot_z_start, rot_z_end, step_rots_z)

    hkls =utils. create_possible_reflections(phase, smallest_number, largest_number)

    list_hkl = []
    list_hkls = []

    if phase == "Monoclinic":
        lattice_structure = utils.Monoclinic_Lattice(initial_crystal_orientation = initial_crystal_orientation)    
    elif phase == "Hexagonal":
        lattice_structure = utils.Hexagonal_Lattice(initial_crystal_orientation = initial_crystal_orientation)

    for rotx in rots_x:
        for rotz in rots_z:
            
            lattice_structure.Apply_Rotation(rotx, rotz, rotation_order = rotation_order)

            Q_hkls = utils.calculate_Q_hkl(hkls, lattice_structure.reciprocal_lattice)

            bla = []
            for w, Q_hkl in enumerate(Q_hkls):
                if utils.check_Bragg_condition(Q_hkl, wavelength, E_bandwidth) == True: 
                    dx, dy, dz = utils.diffraction_direction(Q_hkl, wavelength, sample_detector_distance, tilting_angle)
                    if utils.diffraction_in_detector(dx, dy, dz, detector) == True:
                        list_hkl.append([rotx, rotz, hkls[w][0], hkls[w][1], hkls[w][2]])
                        bla.append([rotx, rotz, hkls[w][0], hkls[w][1], hkls[w][2]])     
                    else:
                        pass
                else:
                    pass

            if not bla:
                pass
            else:
                list_hkls.append(bla)

    list_hkl = np.array(list_hkl)
        
    groups = {}

    for lista in list_hkl:
        hkl = tuple(lista[-3:])
        if hkl not in groups:
            groups[hkl] = [lista]
        else:
            groups[hkl].append(lista)

    x_mesh, y_mesh = np.meshgrid(rots_x, rots_z)

    x_min = min(rots_x)
    x_max = max(rots_x) + 1
    z_min = min(rots_z)
    z_max = max(rots_z) + 1

    cmap = plt.cm.colors.ListedColormap(['white', 'blue'])

    claves = list(groups.keys())
    organ_combinations = [[] for _ in range(len(claves))]
    organ_combinations[0].extend([(clave,) for clave in claves])
    for r in range(2, len(claves) + 1):
        combinations = list(itertools.combinations(claves, r))
        organ_combinations[r - 1] = combinations
    
    plotting = [plot_singles, plot_doubles, plot_triples, plot_fourths, plot_more_than_four]

    for organ_combination in organ_combinations:

        length = len(organ_combination[0])

        if length == 1:
            plot = plotting[0]

        elif length == 2:
            plot = plotting[1]
        
        elif length == 3:
            plot = plotting[2]
        
        elif length == 4:
            plot = plotting[3]
        
        elif length > 4:
            plot = plotting[-1]

        if plot == True:

            for w, combination in enumerate(organ_combination):
                keys_to_compare = combination
                first_values = {}
                common = set()

                for key in keys_to_compare:
                    lista = groups.get(key, [])
                    first_values[key] = set(map(tuple, [arr[:2] for arr in lista]))
                    if not common:
                        common = first_values[key]
                    else:
                        common = common.intersection(first_values[key])

                malla = np.zeros_like(x_mesh)
            
                common = list(common)
                for successfull_condition in common:
                    x_index = np.where(rots_x == successfull_condition[0])
                    z_index = np.where(rots_z == successfull_condition[1])
                    malla[z_index, x_index] = 1
                
                if not common:
                    pass
                else:
                    hkls  = ', '.join([f'({", ".join(map(lambda x: str(int(x)), tup))})' for tup in combination])
                    plt.figure(figsize = (6,6))
                    plt.rcParams.update({'font.size': 15})
                    plt.imshow(malla, extent=[x_min, x_max, z_min, z_max], origin="lower", cmap=cmap, vmin=0, vmax=1)
                    plt.xlabel("Rotation $\perp$ x [°]")
                    plt.ylabel("Rotation $\perp$ z [°]")
                    plt.title("[h,k,l]\n%s"%hkls)
                    plt.tight_layout()
                    plt.grid()
                    plt.show()
        
        else:
            pass
    
def tracking_specific_reflections(phase, detector, sample_detector_distance, wavelength, rot_x_start, rot_x_end, step_rot_x, rot_z_start, rot_z_end, step_rot_z, E_bandwidth, desired_reflections_list, tilting_angle = 0, unrot_axis_value = 0, margin = 0, beam_center = (0,0), savefig = False, fig_name = None, initial_crystal_orientation = None, rotation_order = "xyz"):

    detector = utils.Detector(detector_type=detector, sample_detector_distance=sample_detector_distance, tilting_angle=tilting_angle, margin = margin, beam_center = beam_center)

    step_rots_x = int(abs((rot_x_start - rot_x_end)/step_rot_x)) +1
    rots_x = np.linspace(rot_x_start, rot_x_end, step_rots_x)
    step_rots_z = int(abs((rot_z_start - rot_z_end)/step_rot_z)) +1
    rots_z = np.linspace(rot_z_start, rot_z_end, step_rots_z)

    list_hkl = []

    if phase == "Monoclinic":
        lattice_structure = utils.Monoclinic_Lattice(initial_crystal_orientation = initial_crystal_orientation)        
    elif phase == "Hexagonal":
        lattice_structure = utils.Hexagonal_Lattice(initial_crystal_orientation = initial_crystal_orientation)


    for rotx in rots_x:
        for rotz in rots_z:
            roty = unrot_axis_value

            lattice_structure.Apply_Rotation(rotx, roty, rotz, rotation_order = rotation_order)

            #a,b,c = lattice_structure.crystal_orientation
            #a_rec, b_rec, c_rec = utils.cal_reciprocal_lattice(a, b, c)

            G_hkls = []
            for hkl in desired_reflections_list:
                Q_hkl =  utils.calculate_Q_hkl(hkl, lattice_structure.reciprocal_lattice)

                if utils.check_Bragg_condition(Q_hkl, wavelength, E_bandwidth) == True: 
                    dx, dy, dz = utils.diffraction_direction(Q_hkl, wavelength, sample_detector_distance, tilting_angle)

                    if utils.diffraction_in_detector(dx, dy, dz, detector) == True:
                        G = np.append(Q_hkl, 1)
                    else:
                        G = np.append(Q_hkl, 0)
                else:
                    G = np.append(Q_hkl, 0)
                    
                G_hkls.append(G)
            
            G_hkls = np.array(G_hkls)

            if 0 in G_hkls[:, -1]:
                pass
            else:
                list_hkl.append([rotx, rotz])

    x_mesh, y_mesh = np.meshgrid(rots_x, rots_z)

    x_min = min(rots_x)
    x_max = max(rots_x) + 1
    z_min = min(rots_z)
    z_max = max(rots_z) + 1

    cmap = plt.cm.colors.ListedColormap(['white', 'blue'])

    malla = np.zeros_like(x_mesh)

    for successfull_condition in list_hkl:
        x_index = np.where(rots_x == successfull_condition[0])
        z_index = np.where(rots_z == successfull_condition[1])
        malla[z_index, x_index] = 1

    if not list_hkl:
        pass
    else:
        hkls  = ', '.join([f'[{", ".join(map(lambda x: str(int(x)), tup))}]' for tup in desired_reflections_list])
        plt.figure(figsize = (6,6))
        plt.rcParams.update({'font.size': 15})
        plt.imshow(malla, extent=[x_min, x_max, z_min, z_max], origin="lower", cmap=cmap, vmin=0, vmax=1)
        plt.xlabel("Rotation$\parallel$x  [°]")
        plt.ylabel("Rotation$\parallel$z  [°]")
        plt.title("(h,k,l)\n%s"%hkls.replace("[", "(").replace("]", ")"))
        plt.tight_layout()
        plt.grid()
        plt.show()

        if savefig == True:

            if fig_name is None:
                fig_name ="Reflection_" + hkls.replace(" ","").replace(",","").replace("]","").replace("[","_")[1:]

            plt.savefig(fig_name, dpi = 300)
        
        else:
            pass

#"""
def polycrystalline_sample(phase, detector, angular_step, sample_detector_distance, wavelength, E_bandwidth, smallest_number, largest_number, calculate_intensities = False, tilting_angle = 0, margin = 0, initial_crystal_orientation = None, beam_center = (0,0), rotation_order = "xyz"):

    if phase == "Monoclinic":
        lattice_structure = utils.Monoclinic_Lattice(initial_crystal_orientation = initial_crystal_orientation)
    elif phase == "Hexagonal":
        lattice_structure = utils.Hexagonal_Lattice(initial_crystal_orientation = initial_crystal_orientation)

    if calculate_intensities == True:
        vanadium_atom = utils.Vanadium()
        oxygen_atom = utils.Oxygen()
        intensities = []
    
    detector = utils.Detector(detector_type=detector, sample_detector_distance=sample_detector_distance, tilting_angle=tilting_angle, margin = margin, beam_center = beam_center)

    hkls = utils.create_possible_reflections(phase, smallest_number, largest_number)

    rots = list(range(0, 360, angular_step))

    fig_size = (8*(detector.height/170)*detector.width/detector.height, 8*(detector.height/170))
    plt.figure(figsize = (fig_size[0], fig_size[1]))


    plt.rcParams.update({'font.size': 15})
    if phase == "Hexagonal":
        plt.title("Detector = %s, Hexagonal"%detector.detector_type)
    elif phase == "Monoclinic":
        plt.title("Detector = %s, Monoclinic"%detector.detector_type)
    plt.grid()
    plt.ion()
    plt.xlim(-0.5*detector.width, 0.5*detector.width)
    plt.xlabel("Detector Width [mm]",fontsize = 16)
    plt.ylim(0, detector.height)
    plt.ylabel("Detector Height [mm]",fontsize = 16)

    dys = []
    dzs = []

    for rotx in rots:
        for roty in rots[:1]:
            for rotz in rots:

                lattice_structure.Apply_Rotation(rotx, roty, rotz, rotation_order = rotation_order)

                Q_hkls = utils.calculate_Q_hkl(hkls, lattice_structure.reciprocal_lattice)

                for w, Q_hkl in enumerate(Q_hkls):
                    if utils.check_Bragg_condition(Q_hkl, wavelength, E_bandwidth) == True: 
                        dx, dy, dz = utils.diffraction_direction(Q_hkl, wavelength, sample_detector_distance, tilting_angle)

                        if utils.diffraction_in_detector(dx, dy, dz, detector) == True:
                            
                            if calculate_intensities == True:
                                two_theta = utils.calculate_two_theta_angle_laue(Q_hkl, wavelength)
                                vanadium_atomic_factor = utils.atomic_structure_factor(vanadium_atom.Atomic_Structure_Factor, wavelength, two_theta)
                                oxygen_atomic_factor = utils.atomic_structure_factor(oxygen_atom.Atomic_Structure_Factor, wavelength, two_theta)
                                structure_factor_vanadium_contribution = utils.structure_factor_given_atom(vanadium_atomic_factor, hkls[w], lattice_structure.Vanadium_fractional_position)
                                structure_factor_oxygen_contribution = utils.structure_factor_given_atom(oxygen_atomic_factor, hkls[w], lattice_structure.Oxygen_fractional_position)
                                final_intensity = structure_factor_vanadium_contribution + structure_factor_oxygen_contribution

                                intensities.append(final_intensity.real)

                            else:
                                pass

                            dy = dy + detector.beam_center[0]
                            dz = dz + detector.beam_center[1]

                            dys.append(dy)
                            dzs.append(dz)
                                

    if calculate_intensities == True:

        intensities = np.array(intensities/np.max(intensities))
        scatter = plt.scatter(dys, dzs, color = "blue", s = 8, alpha=0.5)  
        scatter.set_alpha(intensities)

    else:
        plt.scatter(dys, dzs, color = "blue", s = 8) 
    
    plt.scatter(beam_center[0], beam_center[1], label = "Beam Center", marker='x', color='black', s = 100)

    plt.tight_layout()
    plt.show()

#"""

