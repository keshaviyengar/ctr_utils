import numpy as np

'''
This class implements the Goal Tolerance applied during training. The tolerance reduces through training to make it
easier initially to achieve goals and more difficult as the agent learns a policy.
'''


class GoalTolerance(object):
    def __init__(self, initial_tol, final_tol, function_steps, function_type, eval_tol=0.001, evaluation=False):
        self.init_tol = initial_tol
        self.final_tol = final_tol
        self.N_ts = function_steps
        self.function = function_type
        self.evaluation = evaluation
        self.eval_tol = eval_tol
        assert self.function in ['constant', 'linear',
                                 'decay'], 'Not a valid function. Choose constant, linear or decay.'
        if self.function == 'linear':
            self.a = (self.final_tol - self.init_tol) / self.N_ts
            self.b = self.init_tol

        if self.function == 'decay':
            self.a = np.copy(self.init_tol)
            self.r = 1 - np.power((self.final_tol / self.init_tol), 1 / self.N_ts)

        if evaluation:
            self.current_tol = eval_tol
        else:
            self.current_tol = np.copy(initial_tol)
        self.training_step = 0

    def update(self, timestep):
        """
        Update current goal tolerance based on timestep of training.
        :param timestep: Current timestep of training to update goal tolerance.
        """
        # If set_tol is set to zero, update tolerance else use set tolerance.
        if self.evaluation:
            self.current_tol = self.eval_tol
        else:
            if (self.function == 'linear') and (timestep <= self.N_ts):
                self.current_tol = self.a * timestep + self.b
            elif (self.function == 'decay') and (timestep <= self.N_ts):
                self.current_tol = self.a * np.power(1 - self.r, timestep)
            else:
                self.current_tol = self.final_tol
