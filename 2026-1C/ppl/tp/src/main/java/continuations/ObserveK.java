package continuations;

import java.util.ArrayList;

import distributions.Distribution;
import machine.Machine;
import messages.Message;
import messages.ObserveMessage;

public record ObserveK(ArrayList<Object> address) implements Continuation {
    @Override
    public Message executeOn(Machine machine) {
        Float observed_value = (Float) machine.getNextValue();
        Distribution distribution = (Distribution) machine.getNextValue();

        return new ObserveMessage(address, distribution, observed_value, machine);
    }
}