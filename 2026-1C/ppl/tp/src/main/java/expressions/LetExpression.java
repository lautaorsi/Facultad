package expressions;

import java.util.ArrayList;
import java.util.HashMap;

import continuations.Ev;
import continuations.LetK;
import forms.Form;
import forms.FormList;
import machine.Machine;

public class LetExpression extends Expression{

    @Override
    public void evaluate(Machine machine, ArrayList<Form> tail, HashMap environment, ArrayList address ){
        ArrayList<Form> binds = ((FormList) tail.get(0)).elements();
        ArrayList<Form> body = new ArrayList<>(tail.subList(1, tail.size()));

        if(!binds.isEmpty()){
            LetK letK = new LetK(binds, 0, body, environment, address);

            ArrayList newAddress = new ArrayList(address);
            newAddress.add("let");
            newAddress.add(0);
            Ev ev = new Ev(binds.get(1), environment, newAddress);

            machine.pushContinuation(letK);
            machine.pushContinuation(ev);
        }
        else{
            machine.push_body(body, environment, address);
        }
    }
}