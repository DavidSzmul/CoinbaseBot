import numpy as np
import matplotlib.pyplot as plt

from app.library.Agent import Displayer, DQN_Agent
from app.library.Environment import Environment

class Session(object):

    def __init__(self, agent=None, env=None):
        if agent is None or env is None:
            raise ValueError('Agent and Environment have to be defined')
        self.agent = agent
        self.env = env

    def train(self, nb_steps=1e4, nb_steps_varmup=100, delta_train=1, delta_save=0, delta_display=1, verbose=0, name_model = 'test'):
        Disp = Displayer(delta_display=delta_display) # Used to display performances

        # LOOP Episode
        step=0
        episode=0
        while step<=nb_steps:
            episode+=1

            # Restarting episode - Environment
            episode_reward = 0
            episode_loss = 0
            state = self.env.reset()
            done = False
            
            # LOOP Step
            while not done:       
                step+=1
                # Determine action to do     
                action = self.agent.get_action_training(state)
                new_state, reward, done, _ = self.env.step(action)
                transition = (state, action, reward, new_state, done)
                state = new_state
                loss = self.agent.memorize(transition)

                episode_reward += reward
                episode_loss += loss

                if step>nb_steps_varmup and step%delta_train==0:
                    self.agent.fit()
                if done:
                    # print the score and break out of the loop
                    print("episode: {}, Prc_training:, {:.2f}%, score: {}"
                        .format(episode, step/nb_steps*100, episode_reward))
            
            # END Episode   
            # DISPLAY HISTORIC
            if verbose==1:
                Disp.display_historic(self.agent.epsilon, episode_reward, episode_loss) 

            # SAVE MODEL
            if delta_save>0 and not step % delta_save: # and min_reward >= MIN_REWARD:
                self.agent.save_weights(name_model, overwrite=True)
            
        ### END OF EPISODES/TRAINING
        self.env.close()
        # np.save('historic_reward', ep_rewards)
        self.agent.save_weights(name_model,overwrite=True)
        # END OF DISPLAY
        # plt.ioff(), plt.show()

    def test(self, nb_test=1):

        for _ in range(1, nb_test + 1):
            # Restarting episode - Environment
            state = self.env.reset()
            done = False
            
            # LOOP Step
            while not done:       
                # Determine action to do     
                action = self.agent.get_action(state)
                new_state, _, done, _ = self.env.step(action)
                state = new_state
                self.env.render()
            env.close()

if __name__ == '__main__':
    import os, sys, time
    import matplotlib.pyplot as plt

    ### LIBRARIES
    from library.Environment import Environment
    from library.Agent import DQN_Agent

    ### INITIALIZATION
    env = Environment("gym", 'CartPole-v0')
    env = env.getEnv()    
    path_best_Model = 'CartePole_DDQN'

    USE_PER = True
    USE_DOUBLE_DQN = True
    USE_SOFT_SUPDATE = True

    agent = DQN_Agent(env, loading_model=True, name_model=path_best_Model, 
                        use_PER=USE_PER, use_double_dqn=USE_DOUBLE_DQN, use_soft_update=USE_SOFT_SUPDATE)
    # agent = DQN_Agent(env, layers_model=[16, 16], use_PER=USE_PER, use_double_dqn=USE_DOUBLE_DQN, use_soft_update=USE_SOFT_SUPDATE)
    session = Session(agent=agent, env=env)

    ### TRAIN
    session.train(nb_steps=1e4, verbose=1, name_model='CartePole_DDQN', delta_save=100)
    ### TEST
    # print('Finished')
    # session.test(nb_test=2)