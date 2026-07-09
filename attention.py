#在这部分希望实现region=attention(observation,object)后，返回一个region进行后续的处理


Class attention:
    def __init__(self,observation,object=None):
        self.observation = observation
        self.object = object
        if object is None:
            focus=default_focus()
    
    def default_focus(self):
        left=self.observation.left
        
    
        





#
