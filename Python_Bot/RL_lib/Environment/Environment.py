import gym
class Environment(object):

    def __init__(self, env):

        self.library = library
        self.level = level
        self.env = None
        self.associate_env()

    def associate_env(self):
        switcher = {
            "gym": gym.make(self.level),
            "custom": None,
            }
        self.env = switcher.get(self.library, None)

    def getEnv(self):
        return self.env
        
        

        


