package expressions;

import java.util.ArrayList;
import java.util.HashMap;

import continuations.Ev;
import continuations.SampleK;
import forms.Form;
import machine.Machine;

public class SampleExpression extends Expression{

    @Override
    public void evaluate(Machine machine, ArrayList<Form> tail, HashMap environment, ArrayList address ){
        SampleK sampleK = new SampleK(address);

        ArrayList newAddress = new ArrayList(address);
        newAddress.add("d");
        Ev ev = new Ev(tail.get(0), environment, newAddress);

        machine.pushContinuation(sampleK);
        machine.pushContinuation(ev);
    }
}