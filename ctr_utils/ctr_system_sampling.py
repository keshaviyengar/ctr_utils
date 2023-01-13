import numpy as np

# Ranges
# L = [10e-3, 500.0e-3]
# L_c = [10.0e-3, 500.0e-3]
# d_i = [0.1e-3, 2.0e-3]
# d_o = [0.1e-3, 2.0e-3]
# E_I = [5.0e+9, 50.0e+10]
# G_J = [1.0e+10, 30.0e+10]
# x_curv = [1.0, 25.0]
def sample_tube_parameters(tube_parameters_low, tube_parameters_high, num_discrete, num_tubes):
    #tube_parameters_low = {'L': 10e-3, 'L_c': 10.0e-3, 'd_i': 0.1e-3, 'd_o': 1.5e-3, 'E_I': 5.0e+9,
    #                       'G_J': 1.0e+10, 'x_curv': 1.0}
    #tube_parameters_high = {'L': 500e-3, 'L_c': 500.0e-3, 'd_i': 0.6e-3, 'd_o': 2.0e-3, 'E_I': 50.0e+10,
    #                       'G_J': 30.0e+10, 'x_curv': 25.0}
    #num_discrete = 50
    # Constraints:
    # L >= L_c
    # d_i < d_o
    # L_1 >= L_2 >= L_3
    # di_1 >= di_2 >= di_3
    # do_1 >= do_2 >= do_3
    # xcurve_1 >= xcurve_2 >= xcurve_3
    assert num_tubes in [2,3]
    tube_params = {}
    tube_parameters = []
    # Define sample space for each parameter
    E_I_sample_space = np.linspace(tube_parameters_low['stiffness'], tube_parameters_high['stiffness'], num_discrete)
    G_J_sample_space = np.linspace(tube_parameters_low['torsional_stiffness'], tube_parameters_high['torsional_stiffness'], num_discrete)
    L_sample_space = np.linspace(tube_parameters_low['length'], tube_parameters_high['length'], num_discrete)
    L_c_sample_space = np.linspace(tube_parameters_low['length_curved'], tube_parameters_high['length_curved'], num_discrete)
    d_i_sample_space = np.linspace(tube_parameters_low['diameter_inner'], tube_parameters_low['diameter_inner'] + 0.5e-3, num_discrete)
    x_curv_sample_space = np.linspace(tube_parameters_low['x_curvature'], tube_parameters_high['x_curvature'],
                                      num_discrete)
    diameter_diff = 0.4e-3
    tube_sep = 0.1e-3
    # starting at 2nd index to allow for smaller tubes to be sampled for middle and inner tubes
    tube_params['length'] = np.random.choice(L_sample_space[5:])
    # Sample an L_c smaller than L
    tube_params['length_curved'] = np.random.choice(L_c_sample_space[L_c_sample_space <= tube_params['length']])
    tube_params['diameter_inner'] = np.random.choice(d_i_sample_space)
    tube_params['diameter_outer'] = tube_params['diameter_inner'] + diameter_diff
    tube_params['stiffness'] = np.random.choice(E_I_sample_space)
    tube_params['torsional_stiffness'] = np.random.choice(G_J_sample_space)
    tube_params['x_curvature'] = np.random.choice(x_curv_sample_space)
    tube_params['y_curvature'] = 0.0
    # Append as tube 0 parameters
    tube_parameters.append(tube_params)
    # iterate through tubes starting at tube 1
    # TODO: Diameter differences
    for i in range(1, num_tubes):
        tube_params = {}
        if i == 1:
            tube_params['length'] = np.random.choice(L_sample_space[L_sample_space < tube_parameters[i - 1]['length']][1:])
        else:
            tube_params['length'] = np.random.choice(L_sample_space[L_sample_space < tube_parameters[i - 1]['length']])
        tube_params['length_curved'] = np.random.choice(L_c_sample_space[L_c_sample_space <= tube_params['length']])
        tube_params['diameter_inner'] = tube_parameters[i - 1]['diameter_outer'] + tube_sep
        tube_params['diameter_outer'] = tube_params['diameter_inner'] + diameter_diff
        tube_params['stiffness'] = tube_parameters[0]['stiffness']
        tube_params['torsional_stiffness'] = tube_parameters[0]['torsional_stiffness']
        tube_params['x_curvature'] = np.random.choice(
            x_curv_sample_space[x_curv_sample_space <= tube_parameters[i - 1]['x_curvature']])
        tube_params['y_curvature'] = 0.0
        tube_parameters.append(tube_params)

    if num_tubes == 2:
        tubes = ['inner', 'outer']
    else:
     tubes = ['inner', 'middle', 'outer']
    ctr_system = {}
    for i in range(num_tubes):
        ctr_system[tubes[i]] = tube_parameters[i]
    return ctr_system


if __name__ == '__main__':
    tube_parameters_low = {'length': 10e-3, 'length_curved': 10.0e-3, 'diameter_inner': 0.1e-3,
                           'diameter_outer': 0.1e-3, 'stiffness': 5.0e+9,
                           'torsional_stiffness': 1.0e+10, 'x_curvature': 1.0}
    tube_parameters_high = {'length': 500e-3, 'length_curved': 500.0e-3, 'diameter_inner': 0.1e-3 + 2.0e-3,
                            'diameter_outer': 0.1e-3 + 2.0e-3, 'stiffness': 50.0e+10,
                            'torsional_stiffness': 30.0e+10, 'x_curvature': 25.0}
    num_discrete = 50
    ctr_system = sample_tube_parameters(tube_parameters_low, tube_parameters_high, num_discrete)
    print("Final innermost diameter and outermost diameter")
    print(ctr_system['inner']['diameter_inner'])
    print(ctr_system['outer']['diameter_outer'])
    print(ctr_system['outer']['diameter_outer'] - ctr_system['inner']['diameter_inner'])


