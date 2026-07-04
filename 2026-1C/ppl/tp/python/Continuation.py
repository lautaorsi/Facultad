class Continuation:
    
    def handleProcedure(self, machine):
        raise SystemError("Continuation should implement handling procedure")
    


class LetK(Continuation):

    def __init__(self, binds, i, body, environment, addresses):
        self.binds = binds
        self.i = i
        self.body = body
        self.environment = environment
        self.addresses = addresses

    def handleProcedure(self, machine):
        
        value = machine.getValue()

        binds, i = self.binds, self.i
        
        self.environment[binds[2*i]] = value

        if(2*(i+1) < len(binds)):
            newLetK = LetK(binds, i+1, self.body, self.environment, self.addresses)
            newEv= EV(binds[2*(i+1)+1], self.environment, self.addresses + ('let', 2*(i+1)))

            machine.addContinuation(newLetK)
            machine.addContinuation(newEv)
        else:
            machine.push_body()
        
