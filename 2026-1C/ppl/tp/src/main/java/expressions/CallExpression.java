package expressions;

import java.util.ArrayList;
import java.util.HashMap;

import continuations.CallK;
import continuations.Ev;
import forms.Form;
import machine.Machine;

public class CallExpression extends Expression{

    @Override
    public void evaluate(Machine machine, ArrayList<Form> tail, HashMap environment, ArrayList address ){

        int n = tail.size() - 1; // number of actual arguments (excludes the operator)

        CallK callK = new CallK(n, address);
        machine.pushContinuation(callK);

        for(int i = n; i >= 1; i--){
            ArrayList newAddress = new ArrayList(address);
            newAddress.add(i - 1);

            Ev ev = new Ev(tail.get(i), environment, newAddress);
            machine.pushContinuation(ev);
        }

        ArrayList newAddress = new ArrayList(address);
        newAddress.add("fn");

        Ev ev = new Ev(tail.get(0), environment, newAddress);
        machine.pushContinuation(ev);
    }
}