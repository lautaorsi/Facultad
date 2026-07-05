package messages;

import java.util.ArrayList;

import distributions.Distribution;
import machine.Machine;

//USED TO BE OBJECT
public record SampleMessage(ArrayList<Object> address, Distribution distribution, Machine machine) implements Message{

    @Override
    public boolean isSample(){
        return true;
    }

}