package continuations;

import java.util.ArrayList;
import java.util.HashMap;

import expressions.CallExpression;
import expressions.Expression;
import forms.Form;
import forms.FormList;
import forms.FormLiteral;
import forms.FormSymbol;
import machine.Machine;
import messages.ContinueMessage;
import messages.Message;
import utils.Primitives;

public record Ev(Form expression, HashMap environment, ArrayList address) implements Continuation {
    @Override
    public Message executeOn(Machine machine) {
        if(expression instanceof FormSymbol symbol){
            String name = symbol.text();
            if(environment.containsKey(name)){
                machine.pushValue(environment.get(name));
            }
            else if(Primitives.isPrimitive(name)){
                machine.pushValue(Primitives.getPrimitive(name));
            }
            else{
                throw new IllegalArgumentException("Unknown var: " + name);
            }
        }
        else if(expression instanceof FormLiteral literal){
            machine.pushValue(literal.value());
        }
        else if(expression instanceof FormList list){
            Form head = list.get(0);
            String tag = (head instanceof FormSymbol headSymbol) ? headSymbol.text() : null;

            Expression expressionObject = Expression.createExpression(tag);

            ArrayList<Form> args = (expressionObject instanceof CallExpression)
                ? list.elements()
                : new ArrayList<>(list.elements().subList(1, list.size()));

            expressionObject.evaluate(machine, args, environment, address);
        }

        return new ContinueMessage();
    }
}