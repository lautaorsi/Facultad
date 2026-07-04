class Machine:
    def __init__(self, continuationStack, valueStack=None, environment=None, rng=None, log_w=0.0):
        self.continuationStack = list(continuationStack)
        self.valueStack = [] if valueStack is None else list(valueStack)
        self.environment= {} if environment is None else dict(environment)
        self.rng = rng
        self.log_w = log_w

    def fork(self, rng=None):
        return Machine(continuationStack=self.continuationStack.copy(), 
                 valueStack=self.valueStack.copy(),
                 environment=self.environment.copy(),
                 rng = self.rng if rng is None else rng,
                 log_w = self.log_w
                 )
    

    def resume(self):
        continuationStack = self.continuationStack
        valueStack = self.valueStack
        while not continuationStack.empty():
            continuation = continuationStack.pop()

            continuation.handleProcedure(self)



    def handleEv(self, continuation):

        valueStack = self.valueStack
        continuationStack = self.continuationStack 

        expression = continuation.expression()
        continuationEnvironment = continuation.environment()
        address = continuation.address()

        if isinstance(expression, Symbol):
            if continuationEnvironment in self.environment:
                valueStack.append(continuationEnvironment[expression])
            elif is_primitive(expression):
                valueStack.append(PRIMITIVES[expression])
            else:
                raise NameError(expression)
        
        elif not isinstance(expression, list):
            valueStack.append(expression)
        
        else:
            expression.handleProcedure(self)

        

        