class ConstantHedge:

    def __init__(self, holding):
        # Store the number of shares here
        self.holding = float(holding)

        if not math.finite(self.holding):
            raise ValueError("Strategy must be finite")


    def __call__(self, t, history):
        # Return one holding per path.
        pass