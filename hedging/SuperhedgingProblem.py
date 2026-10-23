

class SuperhedgingProblem():
    def __init__(
            self,
            *,
            payoff,
            trading_times
        ):
            self.payoff = payoff
            self.trading_times = trading_times

    def payoffs(self, paths):
        """ Return one payoff per simulated path"""
        self.payoff(paths)

    def trading_gains(self, paths, H):
        pass

    def terminal_wealth(self, paths, x, H):

        pass

    def shortfall(self, paths,x ,H):
        return np.maximum(payoffs(paths) - terminal_wealth(paths,x,H),0)
