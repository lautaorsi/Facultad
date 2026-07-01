import java.util.HashMap;
import java.util.List;

public abstract class Expression{


    public createExpression(String type){
        if(type == "let"){
            return new LetExpression();
        }
        if(type == "if"){
            return new IfExpression();
        }
        if(type == "Fn"){
            return new FnExpression();
        }
        if(type == "sample"){
            return new SampleExpression();
        }
        if(type == "observe"){
            return new ObserveExpression();
        }
        if(type == "call"){
            return new CallExpression();
        }
    }

    
    public abstract void evaluate(Machine machine, List tail, HashMap environment, List address );




    public class LetExpression extends Expression{

        @Override
        public void evaluate(Machine machine, List tail, HashMap environment, List address ){
            binds = environment.get(1);
            body = environment.subList(2, environment.length());
            if(binds != null){
                
                LetK letK = new LetK(binds, 0, body, environment, address);

                List newAddress = new List(address);
                newAddress.add("let", 0);
                Ev ev = new Ev(binds.get(1), environment, newAddress);
                machine.pushInstruction(letK);
                machine.pushInstruction(ev);
            }
            else{
                _push_body(C, body, env, addr);
            }
        }
    }

    public class IfExpression extends Expression{

        @Override
        public void evaluate(Machine machine, List tail, HashMap environment, List address ){
                test = environment.get(1);
                then = environment.get(2);
                els = environment.get(3);

                Ifk ifK = new IfK(then, els, environment, address);

                
                List newAddress = new List(address);
                newAddress.add("test");
                Ev ev = new Ev(test, environment, newAddress);
                
                machine.pushInstruction(ifK);
                machine.pushInstruction(ev);
        }
    }

    public class FnExpression extends Expression{

        @Override
        public void evaluate(Machine machine, List tail, HashMap environment, List address ){
            params = environment.get(2);
            body = environment.subList(3, environment.length());
            
            Closure closure = new Closure(params, body, environment);
            machine.pushValue(closure);
            
        }
    }

    public class SampleExpression extends Expression{

        @Override
        public void evaluate(Machine machine, List tail, HashMap environment, List address ){
            
            SampleK sampleK = new SampleK(address);

            List newAddress = new List(address);
            newAddress.add("d");
            Ev ev = new Ev(environment.get(1), environment, newAddress);

            machine.pushInstruction(sampleK);
            machine.pushInstruction(ev);
        }
    }

    public class ObserveExpression extends Expression{

        @Override
        public void evaluate(Machine machine, List tail, HashMap environment, List address ){
            
            ObserveK observeK = new ObserveK(address);

            List newAddress1 = new List(address);
            List newAddress2 = new List(address);

            newAddress1.add("v");
            newAddress2.add("d");
            Ev ev1 = new Ev(environment.get(2), environment, newAddress1); 
            Ev ev2 = new Ev(environment.get(1), environment, newAddress2);
            
            machine.pushInstruction(observeK);
            machine.pushInstruction(ev1);
            machine.pushInstruction(ev2);
        }
    }

    public class CallExpression extends Expression{

        @Override
        public void evaluate(Machine machine, List tail, HashMap environment, List address ){
            
            CallK callK = new CallK(environment.length() -1 , address);
            machine.pushInstruction(callK);
            
            Ev ev;
            List newAddress;
            for(int i =environment.length() -1; i >= 0; i--){

                newAddress = new List(address);
                newAddress.add(i-1);
                ev = new Ev(environment.get(i), environment, newAddress);
                machine.pushInstruction(ev);
            }
            
            newAddress = new List(address);
            newAddress.add("fn");

            ev = new Ev(environment.get(0), environment, newAddress);
            machine.pushInstruction(ev);
        }
    }

}