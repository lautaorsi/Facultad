package continuations;

import java.util.ArrayList;
import java.util.HashMap;

import forms.Form;
import machine.Machine;
import messages.ContinueMessage;
import messages.Message;

public record IfK(Form thenBranch, Form elseBranch, HashMap environment, ArrayList address) implements Continuation {
    @Override
    public Message executeOn(Machine machine) {
        boolean test = (Boolean) machine.getNextValue();
        Form branch = test ? thenBranch : elseBranch;
        String tag = test ? "then" : "else";

        ArrayList newAddress = new ArrayList(address);
        newAddress.add(tag);

        Ev newEv = new Ev(branch, environment, newAddress);
        machine.pushContinuation(newEv);

        return new ContinueMessage();
    }
}