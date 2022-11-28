class BufferIstance: # where I store datas i need
    def __init__(self):
        self.p_xes = None
        self.reward = None
        self.actions = None
        self.state = None
        self.down = False
        self.xes = None
        self.p_infoes = None
        self.infoes = None


    def set_buffer_new_data(self,data_c, state, reward, down):
        self.p_xs = [x.cpu().data.numpy() for x in data_c.previous_xs]
        self.p_infoes = [info.cpu().data.numpy()
                           for info in data_c.previous_infos]
        self.xes = [x.cpu().data.numpy() for x in data_c.xs]
        self.state = state.cpu().data.numpy()
        self.reward = reward
        self.infoes = [info.cpu().data.numpy() for info in data_c.infos]
        self.down = down
