from stable_baselines3.common.callbacks import BaseCallback
import torch 
import numpy as np

class CustomCallback(BaseCallback):

    
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self,env,parameter, verbose: int = 0):
        super().__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env # type: VecEnv
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # num_timesteps = n_envs * n times env.step() was called
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = {}  # type: Dict[str, Any]
        # self.globals = {}  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger # type: stable_baselines3.common.logger.Logger
        # Sometimes, for event callback, it is useful
        # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.rewards = []
        self.env=env
        self.parameter=parameter
        self.total=[]
        self.total_episodes=[]
        self.value=0

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        self.total.append(self.locals['rewards'][0])
        self.value+=1
        if self.n_calls%1000==0:
            print(f"timesteps {self.n_calls}")
        if self.locals['dones']==True:
            param=self.parameter
            val=[3.92699082, 2.71433605, 5.0893801]
            mass = torch.tensor(val, dtype=torch.float32)
            low = (1-param) * mass
            high = (1+param) * mass 
            new_masses = torch.rand_like(mass) * (high - low) + low
            self.env.sim.model.body_mass[2:] = new_masses.numpy()  #first is world,second torso and then the others
            self.rewards.append(sum(self.total))
            self.total=[]
            self.total_episodes.append(self.value)
            self.value=0

            

            
            
            

        
            
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """

    def plotRewards(self):
        return self.rewards
    