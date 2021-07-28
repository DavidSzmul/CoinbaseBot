from dataclasses import dataclass

from rl_lib.memory.memory import Experience
from rl_lib.agent.agent import Agent
from rl_lib.environment.environment import Environment

@dataclass
class Train_perfs():
    total_reward: float=0   # Total reward during last cycles of training
    total_loss: float=0     # Total loss during last cycles of training
    epsilon: float=0        # Current ratio between exploration vs exploitation

class Executor:

    train_perfs: Train_perfs()

    def reset_perfs(self):
        self.train_perfs = Train_perfs()

    def reset_env(self, env: Environment):
        '''Reset specific environment'''
        env.reset()
        self.reset_perfs()

    def train_cycle(self, agent: Agent, env: Environment) -> bool:
        '''Cycle of training for the agent based on environment'''

        state = env.get_state()
        action = agent.get_action_training(state)
        new_state, reward, done, _ = env.step(action)
        experience = Experience(state, action, new_state, reward, done)
        loss = agent.memorize(experience)
        
        self.train_perfs.total_reward += reward
        self.train_perfs.total_loss += loss
        self.train_perfs.epsilon = agent.epsilon

        return done
    
    def get_train_perfs(self):
        return self.train_perfs

    def execute_cycle(self, agent: Agent, env: Environment) -> bool:
        '''Cycle of normal execution for the agent based on environment'''
        state = env.get_state()
        action = agent.get_action(state)
        _, _, done, _ = env.step(action)
        return done
        


if __name__=="__main__":
    from rl_lib.agent.dqn import DQN_Agent
    from rl_lib.environment.environment import Default_Env
    exec = Executor()
    agent = DQN_Agent()
    env = Default_Env()

    env.reset()
    exec.train_cycle(agent, env)
    exec.execute_cycle(agent, env)