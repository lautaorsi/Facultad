package instructions.continuations;

import machine.Machine;
import messages.ContinueMessage;
import messages.Message;

public record Discard() implements Continuation {
    @Override
    public Message executeOn(Machine machine) {
        
        machine.getNextValue();
        
        return new ContinueMessage();
    }
}