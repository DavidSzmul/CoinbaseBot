from collections import deque
from RL_lib.Memory.Memory import Experience
from RL_lib.Agent.agent import Agent
from RL_lib.Environment.Environment import Environment

class Executor:

    def __init__(self, len_loss: int=1):
        self.loss_queue = deque(maxlen=len_loss)       

    def reset_env(env: Environment):
        '''Reset specific environment'''
        env.reset()

    def train_cycle(self, agent: Agent, env: Environment) -> bool:

        state = env.get_state()
        action = agent.get_action_training(state)
        new_state, reward, done, _ = env.step(action)
        experience = Experience(state, action, new_state, reward, done)
        loss = agent.memorize(experience)
        self.loss_queue.append(loss)
        return done

    def execute_cycle(self, agent: Agent, env: Environment) -> bool:

        state = env.get_state()
        action = agent.get_action(state)
        _, _, done, _ = env.step(action)
        return done
        


if __name__=="__main__":
    from RL_lib.Agent.dqn import DQN_Agent
    from RL_lib.Environment.Environment import Default_Env
    exec = Executor()
    agent = DQN_Agent()
    env = Default_Env()

    env.reset()
    exec.train_cycle(agent, env)
    exec.execute_cycle(agent, env)