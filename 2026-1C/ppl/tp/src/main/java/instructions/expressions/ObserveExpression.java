package instructions.expressions;

import java.util.ArrayList;
import java.util.HashMap;

import forms.Form;
import instructions.continuations.Ev;
import instructions.continuations.ObserveK;
import machine.Machine;

public class ObserveExpression extends Expression{

    @Override
    public void evaluate(Machine machine, ArrayList<Form> tail, HashMap<String,Object> environment, ArrayList<Object> address ){
        ObserveK observeK = new ObserveK(address);

        ArrayList<Object> newAddressV = new ArrayList<Object>(address);
        newAddressV.add("v");
        ArrayList<Object> newAddressD = new ArrayList<Object>(address);
        newAddressD.add("d");

        Ev evV = new Ev(tail.get(1), environment, newAddressV);
        Ev evD = new Ev(tail.get(0), environment, newAddressD);

        machine.pushContinuation(observeK);
        machine.pushContinuation(evV);
        machine.pushContinuation(evD);
    }
}