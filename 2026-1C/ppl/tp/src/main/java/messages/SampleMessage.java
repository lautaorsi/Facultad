package messages;

import java.util.ArrayList;

import distributions.Distribution;
import machine.Machine;

public record SampleMessage(Object address, Distribution distribution, Machine machine) implements Message{

    @Override
    public boolean isSample(){
        return true;
    }

}