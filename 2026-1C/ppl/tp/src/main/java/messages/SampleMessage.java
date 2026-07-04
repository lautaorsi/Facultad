package messages;

import java.util.ArrayList;

import distributions.Distribution;
import machine.Machine;

public record SampleMessage(ArrayList address, Distribution distribution, Machine machine) implements Message{

    @Override
    public boolean isSample(){
        return true;
    }

}