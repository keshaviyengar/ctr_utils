import numpy as np


def get_obs(joints, desired_goal, achieved_goal, goal_tolerance, min_max_goal_tolerance,
            tube_length):
    """
    :param joints: CTR joints represented as beta_0, beta_1, beta_2, alpha_0, alpha_1, alpha_2
    :param desired_goal: The desired goal of the current episodes
    :param achieved_goal: End-effector position of the robot
    :param goal_tolerance: Current goal tolerance of episode
    :return: Observation of robot.
    """
    num_tubes = 3
    # TODO: Min and max delta goal assumption
    min_max_delta_goals = np.array([[-0.5, -0.5, 0.0], [0.5, 0.5, 1.0]])
    # Convert joints to egocentric normalized representation
    ego_joints = prop2ego(joints, num_tubes)
    ego_joints[num_tubes:] = B_to_B_U(ego_joints[num_tubes:], tube_length[0], tube_length[1], tube_length[2])
    joint_rep = joint2rep(ego_joints, num_tubes)

    # Normalize desired and achieve goals
    norm_dg = normalize(min_max_delta_goals[0], min_max_delta_goals[1], desired_goal)
    norm_ag = normalize(min_max_delta_goals[0], min_max_delta_goals[1], achieved_goal)
    # Normalize goal tolerance
    norm_tol = np.array([normalize(min_max_goal_tolerance[0], min_max_goal_tolerance[1], goal_tolerance)])
    # Concatenate all and return
    return np.concatenate((joint_rep, norm_dg - norm_ag, norm_tol))


def normalize(x_min, x_max, x):
    """
    Normalize input data with maximum and minimum to -1 and 1
    :param x_min: Maximum x value.
    :param x_max: Minimum x value.
    :param x: Input value.
    :return: return normalized x value.
    """
    if type(x) == np.ndarray:
        return 2 * np.divide(x - x_min, x_max - x_min) - 1
    else:
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
    num_tubes = 3
    action[:num_tubes] = action[:num_tubes] * extension_limit
    action[num_tubes:] = action[num_tubes:] * rotation_limit
    joints_low = np.array([-tube_length[0], -tube_length[1], -tube_length[2], -np.inf, -np.inf, -np.inf])
    joints_high = np.array([0.0, 0.0, 0.0, np.inf, np.inf, np.inf])
    # Apply action and ensure within joint limits
    new_joints = np.clip(np.around(joints + action, 6), joints_low, joints_high)
    # Check if extension joints are not colliding
    betas_U = B_to_B_U(new_joints[:num_tubes], tube_length[0], tube_length[1], tube_length[2])
    # Check if beta extension are within limits, if not, return old joints
    if not np.any(betas_U < -1.0) and not np.any(betas_U > 1.0):
        return new_joints
    else:
        return joints


def sample_goal(tube_length):
    betas = B_U_to_B(np.random.uniform(low=-np.ones(3), high=np.ones(3)), tube_length[0],
                     tube_length[1], tube_length[2])
    alphas = alpha_U_to_alpha(np.random.uniform(low=-np.ones(3), high=np.ones(3)), np.pi)
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


def rep2joint(rep, num_tubes):
    """
    Convert trigonometric representation of all tubes to simple joint representation.
    :param num_tubes: Number of tubes for robot
    :param rep: Trigonometric representation of all tubes as array.
    :return: Simple joint representation [beta_0, ..., beta_2, alpha_0, ..., alpha_2]
    """
    rep = [rep[i:i + num_tubes] for i in range(0, len(rep), num_tubes)]
    beta = np.empty(num_tubes)
    alpha = np.empty(num_tubes)
    for tube in range(0, num_tubes):
        joint = single_trig2joint(rep[tube])
        beta[tube] = joint[0]
        alpha[tube] = joint[1]
    return np.concatenate((beta, alpha))


def joint2rep(joint, num_tubes):
    """
    Convert simple joint representation to trigonometric representation.
    :param joint: Simple joints as [beta_0, ..., beta_2, alpha_0, ..., alpha_2]
    :param num_tubes: Number of tubes for robot
    :return: Trigonoetric representation of all tubes [(cos(alpha_0), sin(alpha_0), beta_0), ... (cos(alpha_2), sin(alpha_2), beta_2a)]
    """
    rep = np.array([])
    betas = joint[:num_tubes]
    alphas = joint[num_tubes:]
    for beta, alpha in zip(betas, alphas):
        trig = single_joint2trig(np.array([beta, alpha]))
        rep = np.append(rep, trig)
    return rep


def ego2prop(joint, num_tubes):
    """
    Convert from egocentric joint representation to proprioceptive representation
    :param joint: Input joints are [beta_0, ..., beta_2, alpha_0, ..., alpha_2]_ego
    :param num_tubes: Number of tubes for robot
    :return:[beta_0, ..., beta_2, alpha_0, ..., alpha_2]_prop joint representation
    """
    rel_beta = joint[:num_tubes]
    rel_alpha = joint[num_tubes:]
    betas = rel_beta.cumsum()
    alphas = rel_alpha.cumsum()
    return np.concatenate((betas, alphas))


def prop2ego(joint, num_tubes):
    """
    Convert from proprioceptive joint representation to egocentric representation
    :param joint: Input joints are [beta_0, ..., beta_2, alpha_0, ..., alpha_2]_prop
    :param num_tubes: Number of tubes for robot
    :return:[beta_0, ..., beta_2, alpha_0, ..., alpha_2]_ego joint representation
    """
    betas = joint[:num_tubes]
    alphas = joint[num_tubes:]
    # Compute difference
    rel_beta = np.diff(betas, prepend=0)
    rel_alpha = np.diff(alphas, prepend=0)
    return np.concatenate((rel_beta, rel_alpha))


# TODO: Consider a two tube system as well
# Conversion between normalized and un-normalized joints
def B_U_to_B(B_U, L_1, L_2, L_3):
    B_U = np.append(B_U, 1)
    M_B = np.array([[-L_1, 0, 0],
                    [-L_1, L_1 - L_2, 0],
                    [-L_1, L_1 - L_2, L_2 - L_3]])
    normalized_B = np.block([[0.5 * M_B, 0.5 * M_B @ np.ones((3, 1))],
                             [np.zeros((1, 3)), 1]])
    B = normalized_B @ B_U
    return B[:3]


# TODO: Consider a two tube system as well
def B_to_B_U(B, L_1, L_2, L_3):
    B = np.append(B, 1)
    M_B = np.array([[-L_1, 0, 0],
                    [-L_1, L_1 - L_2, 0],
                    [-L_1, L_1 - L_2, L_2 - L_3]])
    normalized_B = np.block([[0.5 * M_B, 0.5 * M_B @ np.ones((3, 1))],
                             [np.zeros((1, 3)), 1]])
    B_U = np.around(np.linalg.inv(normalized_B) @ B, 6)
    return B_U[:3]


def alpha_U_to_alpha(alpha_U, alpha_max):
    return alpha_max * alpha_U


def alpha_to_alpha_U(alpha, alpha_max):
    return 1. / alpha_max * alpha


def sample_joints(tube_length):
    betas = B_U_to_B(np.random.uniform(low=-np.ones(3), high=np.ones(3)), tube_length[0],
                     tube_length[1], tube_length[2])
    alphas = alpha_U_to_alpha(np.random.uniform(low=-np.ones(3), high=np.ones(3)), np.pi)
    return np.concatenate((betas, alphas))

def flip_joints(joints):
    num_tubes = 3
    betas = joints[:num_tubes]
    alphas = joints[num_tubes:]
    return np.concatenate((betas, alphas))


if __name__ == '__main__':
    B = np.array([[0.0, 0.0, -0.05]])
    B_U = B_to_B_U(B, 0.11, 0.165, 0.21)
