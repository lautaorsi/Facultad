package continuations;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;

import forms.FormSymbol;
import machine.Machine;
import messages.ContinueMessage;
import messages.Message;
import utils.Closure;
import utils.Primitives.Primitive;

public record CallK(int n, ArrayList<Object> address) implements Continuation {
    @Override
    public Message executeOn(Machine machine) {

        ArrayList<Object> arguments = new ArrayList<>();

        for(int i = 0; i < n; i++){
            arguments.add(machine.getNextValue());
        }
        Collections.reverse(arguments);

        Object f = machine.getNextValue();

        if(f instanceof Closure closure){
            HashMap<String,Object> newEnv = new HashMap<String,Object>(closure.environment());

            for(int i = 0; i < closure.parameters().size(); i++){
                String paramName = ((FormSymbol) closure.parameters().get(i)).text();
                newEnv.put(paramName, arguments.get(i));
            }

            machine.push_body(closure.body(), newEnv, address);
        }
        else{
            machine.pushValue(((Primitive) f).apply(arguments));
        }

        return new ContinueMessage();
    }
}