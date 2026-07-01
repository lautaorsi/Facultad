import java.beans.Expression;
import java.util.Dictionary;
import java.util.HashMap;
import java.util.List;

public interface Continuation{
        void executeOn(Machine m);
}

public record Ev(Expression expression, HashMap environment, List address) implements Continuation {
    @Override
    public void executeOn(Machine m) {
        m.handleEv(this); 
    }
}

public record LetK(List binds, Int i, List body, HashMap environment, List address) implements Continuation {
    @Override
    public void executeOn(Machine m) {
        m.handleLetK(this); 
    }
}

public record IfK(T2 thenBranch, T3 elseBranch, HashMap environment, List address) implements Continuation {
    @Override
    public void executeOn(Machine m) {
        m.handleIfK(this); 
    }
}

public record Discard() implements Continuation {
    @Override
    public void executeOn(Machine m) {
        m.handleDiscard(this); 
    }
}

public record CallK(Int n, List address) implements Continuation {
    @Override
    public void executeOn(Machine m) {
        m.handleCallK(this); 
    }
}

public record SampleK(List addres) implements Continuation {
    @Override
    public void executeOn(Machine m) {
        m.handleSampleK(this); 
    }
}

public record ObserveK(List addres) implements Continuation {
    @Override
    public void executeOn(Machine m) {
        address = self.address;
        y = m.getNextValue();
        d = m.getNextValue();
        controller.sendObserveRequest(address, d, y, m); 
    }
}








    public void handleEv(Continuation continuation){
        expression = continuation.expression;
        environment = continuation.environment;
        address = continuation.address;
                if(e instanceof Symbol){
                    if(environment.containsKey(e)){
                        self.pushValue(env.get(e));
                    } 
                    else{
                        if(is_primitive(e)){
                            V.add(PRIMITIVES[e]);
                        }
                        else{
                            throw new IllegalArgumentException("Unknown var: " + name);
                        }
                    }
                    
                }
                else{
                    if(! e instanceof List){
                        self.pushValue(e);
                    }
                    else{
                        head = e[0];
                        tail = e.subList(1, e.size());
                        Expression expressionObject = new createExpression(head); 

                        expressionObject.evaluate(self, tail, environment, address);
                    }
                }
    }

    public void handleLetK( Continuation continuation){
        binds = continuation.binds;
        i = continuation.i;
        body = continuation.body;
        environment = continuation.environment;
        address = continuation.address;
        environment.put(binds[2*i], self.getNextValue());
        if( 2*(i+1) < len(binds)){
            LetK newLetK = new LetK(bins, i+1, body, environment, address);
            self.pushInstruction(newLetK);
            
            Ev newEv = new Ev(binds[2*(i+1)+1], environment, address);
            self.pushInstruction(newEv);

            List newAddress = new List(address);
            newAddress.add("let",2*(i+1));
            newEv = new Ev(binds[2*(i+1)+1], environment, newAddress);
            self.pushInstruction(newEv);
        
        }
        else{
            self.push_body(body, environment, address);
        }
    }

    public void handleIfK( Continuation continuation){
                then = continuation.thenBranch;
                els = continuation.elseBranch;
                environment = continuation.environment;
                address = continuation.address;

                branch = (self.getNextValue()) ? then : els; 
                tag = (self.getNextValue()) ? "then" : "else";
                
                List newAddress = new List();
                newAddress.add(tag);
                Ev newEv = new Ev(branch, environment, newAddress);
                self.pushInstruction(newEv);
    }

    public void handleDiscard( Continuation continuation){
        self.getNextValue();
    }

    public void handleCallK( Continuation continuation){
                n = continuation.n;
                addr = continuation.address;

                List<E> arguments = new List();

                for(int i = 0; i < n; i++){
                    arguments.add(self.getNextValue());
                }

                arguments = arguments.reversed();
                f = self.getNextValue();
                
                if(f instanceof Closure){
                    Dictionary new_env = f.environment;
                    for(int i = 0; i < f.params.length(); i++){
                        new_env[f.params[i]] = arguments[i];
                        self.push_body(f.body, new_env, addr);
                    }
                }
                else{
                    self.addValue(f.apply(args));
                }
    }

    public void handleSampleK( Continuation continuation){
        address = continuation.address;
        d = self.getNextValue();
        
        controller.sendSampleRequest(address, distribution, self);
    }

    public Continuation handleObserveK( Continuation continuation){
                address = continuation.address;
                y = self.getNextValue();
                d = self.getNextValue();
                controller.sendObserveRequest(addr, d, y, m);
    }
