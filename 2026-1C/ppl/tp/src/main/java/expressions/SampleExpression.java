package expressions;

import java.util.ArrayList;
import java.util.HashMap;

import continuations.Ev;
import continuations.SampleK;
import forms.Form;
import machine.Machine;

public class SampleExpression extends Expression{

    @Override
    public void evaluate(Machine machine, ArrayList<Form> tail, HashMap<String,Object> environment, ArrayList<Object> address ){
        SampleK sampleK = new SampleK(address);

        ArrayList<Object> newAddress = new ArrayList<Object>(address);
        newAddress.add("d");
        Ev ev = new Ev(tail.get(0), environment, newAddress);

        machine.pushContinuation(sampleK);
        machine.pushContinuation(ev);
    }
}