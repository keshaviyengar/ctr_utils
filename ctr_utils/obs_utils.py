import numpy as np
from collections import OrderedDict

KEY_ORDER = ['observation', 'achieved_goal', 'desired_goal']


def get_obs(joints, joint_representation, desired_goal, achieved_goal, goal_tolerance, min_max_goal_tolerance,
            tube_length):
    """
    :param tube_length:
    :param min_max_goal_tolerance:
    :param joint_representation:
    :param joints: CTR joints represented as beta_0, beta_1, beta_2, alpha_0, alpha_1, alpha_2
    :param desired_goal: The desired goal of the current episodes
    :param achieved_goal: End-effector position of the robot
    :param goal_tolerance: Current goal tolerance of episode
    :return: Observation of robot.
    """
    num_tubes = len(tube_length)
    assert num_tubes in [2, 3]
    # TODO: Min and max delta goal assumption
    x_y_max = 2.0
    z_max = 4.0
    min_max_delta_goals = np.array([[-x_y_max, -x_y_max, 0.0], [x_y_max, x_y_max, z_max]])
    # Convert joints to egocentric representation
    joints = np.copy(joints)
    if joint_representation == 'egocentric':
        joints = prop2ego(joints, num_tubes)

    if num_tubes == 2:
        joints[:num_tubes] = B_to_B_U(joints[:num_tubes], tube_length[0], tube_length[1])
    else:
        joints[:num_tubes] = B_to_B_U(joints[:num_tubes], tube_length[0], tube_length[1], tube_length[2])
    joint_rep = joint2rep(joints)

    # Normalize desired and achieve goals
    norm_dg = normalize(min_max_delta_goals[0], min_max_delta_goals[1], desired_goal)
    norm_ag = normalize(min_max_delta_goals[0], min_max_delta_goals[1], achieved_goal)
    # Normalize goal tolerance
    norm_tol = np.array([normalize(min_max_goal_tolerance[0], min_max_goal_tolerance[1], goal_tolerance)])
    # Concatenate all and return
    return np.concatenate((joint_rep, norm_dg - norm_ag, norm_tol))


def convert_dict_to_obs(obs_dict):
    """
    :param obs_dict: (dict<np.ndarray>)
    :return: (np.ndarray)
    """
    return np.concatenate([obs_dict[key] for key in KEY_ORDER])


def convert_obs_to_dict(observations, achieved_goal, desired_goal):
    """
    Inverse operation of convert_dict_to_obs
    :param observations: (np.ndarray)
    :return: (OrderedDict<np.ndarray>)
    """
    return OrderedDict([
        ('observation', observations),
        ('achieved_goal', achieved_goal),
        ('desired_goal', desired_goal),
    ])


def normalize(x_min, x_max, x):
    """
    Normalize input data with maximum and minimum to -1 and 1
    :param x_min: Maximum x value.
    :param x_max: Minimum x value.
    :param x: Input value.
    :return: return normalized x value.
    """
    if type(x) == list:
        print("x is a list, please use np.ndarrays. Casting...")
        x = np.array(x)
    if type(x) == np.ndarray:
        assert np.any(x_min != x_max), "x_min and and x_max are equal. Will cause divide by zero error."
        assert np.any(x <= x_max), "Values larger than x_max"
        assert np.any(x >= x_min), "Values smaller than x_min"
        return 2 * np.divide(x - x_min, x_max - x_min) - 1
    else:
        assert x_min != x_max, "x_min and and x_max are equal. Will cause divide by zero error."
        assert x <= x_max, "Values larger than x_max"
        assert x >= x_min, "Values smaller than x_min"
        return 2 * (x - x_min) / (x_max - x_min) - 1


def apply_action(action, extension_limit, rotation_limit, joints, tube_length):
    """
    :param action: Selected change in joint values by agent
    :param joints: Joint configuration of robot
    :param joint_limits_low: Low of joint space
    :param joint_limits_high: High of joint space
    :param tube_length: Tube lengths (outermost to innermost)
    :return: joints returned
    """
    num_tubes = int(len(action) / 2)
    assert num_tubes in [2, 3]
    assert len(tube_length) == num_tubes

    joints = np.copy(joints)
    action[:num_tubes] = action[:num_tubes] * extension_limit
    action[num_tubes:] = action[num_tubes:] * rotation_limit
    if num_tubes == 2:
        joints_low = np.array([-tube_length[0], -tube_length[1], -np.inf, -np.inf])
        joints_high = np.array([0.0, 0.0, np.inf, np.inf])
    else:
        joints_low = np.array([-tube_length[0], -tube_length[1], -tube_length[2], -np.inf, -np.inf, -np.inf])
        joints_high = np.array([0.0, 0.0, 0.0, np.inf, np.inf, np.inf])
    # Apply action and ensure within joint limits
    new_joints = np.clip(joints + action, joints_low, joints_high)
    # Check if extension joints are not colliding
    if num_tubes == 2:
        betas_U = B_to_B_U(new_joints[:num_tubes], tube_length[0], tube_length[1])
    else:
        betas_U = B_to_B_U(new_joints[:num_tubes], tube_length[0], tube_length[1], tube_length[2])
    # Check if beta extension are within limits, if not, return old joints
    if not np.any(betas_U < -1.0) and not np.any(betas_U > 1.0):
        return new_joints
    else:
        return joints


def sample_goal(tube_length):
    num_tubes = len(tube_length)
    assert num_tubes in [2, 3]
    betas = B_U_to_B(np.random.uniform(low=-np.ones(num_tubes), high=np.ones(num_tubes)), tuple(tube_length))
    alphas = alpha_U_to_alpha(np.random.uniform(low=-np.ones(num_tubes), high=np.ones(num_tubes)), np.pi)
    return np.concatenate((betas, alphas))


def single_joint2trig(joint):
    """
    Converting single tube extension and rotation to trigonometric representation.
    :param joint: Joint values of single tube [beta, alpha]
    :return: Trigonometric representation (cos(alpha), sin(alpha), beta)
    """
    return np.array([np.cos(joint[1]), np.sin(joint[1]), joint[0]])


def single_trig2joint(trig):
    """
    Converting single tube trigonometric representation to joint extension and rotation.
    :param trig: Input trigonometric representation as (cos(alpha), sin(alpha), beta)
    :return: return [beta, alpha]
    """
    return np.array([trig[2], np.arctan2(trig[1], trig[0])])


def rep2joint(rep):
    """
    Convert trigonometric representation of all tubes to simple joint representation.
    :param num_tubes: Number of tubes for robot
    :param rep: Trigonometric representation of all tubes as array.
    :return: Simple joint representation [beta_0, ..., beta_2, alpha_0, ..., alpha_2]
    """
    num_tubes = int(len(rep) / 3)
    rep = [rep[i:i + num_tubes] for i in range(0, len(rep), num_tubes)]
    beta = np.empty(num_tubes)
    alpha = np.empty(num_tubes)
    for tube in range(0, num_tubes):
        joint = single_trig2joint(rep[tube])
        beta[tube] = joint[0]
        alpha[tube] = joint[1]
    return np.concatenate((beta, alpha))


def joint2rep(joint):
    """
    Convert simple joint representation to trigonometric representation.
    :param joint: Simple joints as [beta_0, ..., beta_2, alpha_0, ..., alpha_2]
    :param num_tubes: Number of tubes for robot
    :return: Trigonoetric representation of all tubes [(cos(alpha_0), sin(alpha_0), beta_0), ... (cos(alpha_2), sin(alpha_2), beta_2a)]
    """
    num_tubes = int(len(joint) / 2)
    assert num_tubes in [2,3]
    rep = np.array([])
    betas = joint[:num_tubes]
    alphas = joint[num_tubes:]
    for beta, alpha in zip(betas, alphas):
        trig = single_joint2trig(np.array([beta, alpha]))
        rep = np.append(rep, trig)
    return rep


def ego2prop(joint):
    """
    Convert from egocentric joint representation to proprioceptive representation
    :param joint: Input joints are [beta_0, ..., beta_2, alpha_0, ..., alpha_2]_ego
    :param num_tubes: Number of tubes for robot
    :return:[beta_0, ..., beta_2, alpha_0, ..., alpha_2]_prop joint representation
    """
    num_tubes = int(len(joint) / 2)
    assert num_tubes in [2,3]
    rel_beta = joint[:num_tubes]
    rel_alpha = joint[num_tubes:]
    betas = rel_beta.cumsum()
    alphas = rel_alpha.cumsum()
    return np.concatenate((betas, alphas))


def prop2ego(joint):
    """
    Convert from proprioceptive joint representation to egocentric representation
    :param joint: Input joints are [beta_0, ..., beta_2, alpha_0, ..., alpha_2]_prop
    :param num_tubes: Number of tubes for robot
    :return:[beta_0, ..., beta_2, alpha_0, ..., alpha_2]_ego joint representation
    """
    num_tubes = int(len(joint) / 2)
    assert num_tubes in [2,3]
    betas = joint[:num_tubes]
    alphas = joint[num_tubes:]
    # Compute difference
    rel_beta = np.diff(betas, prepend=0)
    rel_alpha = np.diff(alphas, prepend=0)
    return np.concatenate((rel_beta, rel_alpha))


## Conversion between normalized and un-normalized joints
#def B_U_to_B(B_U, L_1, L_2, L_3):
#    B_U = np.append(B_U, 1)
#    M_B = np.array([[-L_1, 0, 0],
#                    [-L_1, L_1 - L_2, 0],
#                    [-L_1, L_1 - L_2, L_2 - L_3]])
#    normalized_B = np.block([[0.5 * M_B, 0.5 * M_B @ np.ones((3, 1))],
#                             [np.zeros((1, 3)), 1]])
#    B = normalized_B @ B_U
#    return B[:3]

# Conversion between normalized and un-normalized joints
def B_U_to_B(B_U, *L_args):
    # Ensure number of tubes is either 2 or 3
    assert len(L_args) in [2, 3]
    num_tubes = len(L_args)
    if num_tubes == 2:
        (L_1, L_2) = L_args
        B_U = np.append(B_U, 1)
        M_B = np.array([[-L_1, 0],
                        [-L_1, L_1 - L_2]])
    else:
        (L_1, L_2, L_3) = L_args
        B_U = np.append(B_U, 1)
        M_B = np.array([[-L_1, 0, 0],
                        [-L_1, L_1 - L_2, 0],
                        [-L_1, L_1 - L_2, L_2 - L_3]])

    normalized_B = np.block([[0.5 * M_B, 0.5 * M_B @ np.ones((num_tubes, 1))],
                             [np.zeros((1, num_tubes)), 1]])
    B = normalized_B @ B_U
    return B[:num_tubes]


#def B_to_B_U(B, L_1, L_2, L_3):
#    B = np.append(B, 1)
#    M_B = np.array([[-L_1, 0, 0],
#                    [-L_1, L_1 - L_2, 0],
#                    [-L_1, L_1 - L_2, L_2 - L_3]])
#    normalized_B = np.block([[0.5 * M_B, 0.5 * M_B @ np.ones((3, 1))],
#                             [np.zeros((1, 3)), 1]])
#    B_U = np.around(np.linalg.inv(normalized_B) @ B, 6)
#    return B_U[:3]

def B_to_B_U(B, *L_args):
    # Ensure number of tubes is either 2 or 3
    assert len(L_args) in [2, 3]
    num_tubes = len(L_args)
    B = np.append(B, 1)
    if num_tubes == 2:
        (L_1, L_2) = L_args
        M_B = np.array([[-L_1, 0],
                        [-L_1, L_1 - L_2]])

    else:
        (L_1, L_2, L_3) = L_args
        M_B = np.array([[-L_1, 0, 0],
                        [-L_1, L_1 - L_2, 0],
                        [-L_1, L_1 - L_2, L_2 - L_3]])

    normalized_B = np.block([[0.5 * M_B, 0.5 * M_B @ np.ones((num_tubes, 1))],
                            [np.zeros((1, num_tubes)), 1]])
    B_U = np.around(np.linalg.inv(normalized_B) @ B, 6)
    return B_U[:num_tubes]


def alpha_U_to_alpha(alpha_U, alpha_max):
    return alpha_max * alpha_U


def alpha_to_alpha_U(alpha, alpha_max):
    return 1. / alpha_max * alpha


def sample_joints(tube_length):
    num_tubes = len(tube_length)
    assert num_tubes in [2, 3]
    if num_tubes == 2:
        betas = B_U_to_B(np.random.uniform(low=-np.ones(num_tubes), high=np.ones(num_tubes)), tube_length[0],
                         tube_length[1])
    else:
        betas = B_U_to_B(np.random.uniform(low=-np.ones(num_tubes), high=np.ones(num_tubes)), tube_length[0],
                         tube_length[1], tube_length[2])
    alphas = alpha_U_to_alpha(np.random.uniform(low=-np.ones(num_tubes), high=np.ones(num_tubes)), np.pi)
    return np.concatenate((betas, alphas))


def flip_joints(joints):
    num_tubes = int(len(joints) / 2)
    assert num_tubes in [2, 3]
    betas = np.flip(joints[:num_tubes])
    alphas = np.flip(joints[num_tubes:])
    return np.concatenate((betas, alphas))


if __name__ == '__main__':
    B = np.array([[0.0, 0.0, -0.05]])
    B_U = B_to_B_U(B, 0.11, 0.165, 0.21)
