package continuations;

import java.util.ArrayList;

import distributions.Distribution;
import machine.Machine;
import messages.Message;
import messages.SampleMessage;


public record SampleK(ArrayList<Object> address) implements Continuation {
    @Override
    public Message executeOn(Machine machine) {
        Distribution distribution = (Distribution) machine.getNextValue();


        return new SampleMessage(address, distribution, machine);
    }
}