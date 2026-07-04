package expressions;

import java.util.ArrayList;
import java.util.HashMap;

import continuations.Ev;
import continuations.ObserveK;
import forms.Form;
import machine.Machine;

public class ObserveExpression extends Expression{

    @Override
    public void evaluate(Machine machine, ArrayList<Form> tail, HashMap environment, ArrayList address ){
        ObserveK observeK = new ObserveK(address);

        ArrayList newAddressV = new ArrayList(address);
        newAddressV.add("v");
        ArrayList newAddressD = new ArrayList(address);
        newAddressD.add("d");

        Ev evV = new Ev(tail.get(1), environment, newAddressV);
        Ev evD = new Ev(tail.get(0), environment, newAddressD);

        machine.pushContinuation(observeK);
        machine.pushContinuation(evV);
        machine.pushContinuation(evD);
    }
}