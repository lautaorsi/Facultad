package continuations;

import java.util.ArrayList;
import java.util.HashMap;

import forms.Form;
import forms.FormSymbol;
import machine.Machine;
import messages.ContinueMessage;
import messages.Message;

public record LetK(ArrayList<Form> binds, int i, ArrayList<Form> body, HashMap environment, ArrayList address) implements Continuation {
    @Override
    public Message executeOn(Machine machine) {
        String key = ((FormSymbol) binds.get(2 * i)).text();


        HashMap newEnv = new HashMap(environment);
        newEnv.put(key, machine.getNextValue());
        
        
        if (2 * (i + 1) < binds.size()) {
            LetK newLetK = new LetK(binds, i + 1, body, newEnv, address);
            machine.pushContinuation(newLetK);

            ArrayList newAddress = new ArrayList(address);
            newAddress.add("let");
            newAddress.add(2 * (i + 1));

            Ev newEv = new Ev(binds.get(2 * (i + 1) + 1), newEnv, newAddress);
            machine.pushContinuation(newEv);
        }
        else {
            machine.push_body(body, newEnv, address);
        }

        
        return new ContinueMessage();
    }
}