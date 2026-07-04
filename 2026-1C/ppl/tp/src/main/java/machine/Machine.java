package machine;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

import continuations.Continuation;
import continuations.Discard;
import continuations.Ev;
import forms.Form;
import forms.FormList;
import forms.FormSymbol;
import messages.ContinueMessage;
import messages.DoneMessage;
import messages.Message;
import utils.Closure;
import utils.Parser;

public class Machine{

    private ArrayDeque<Continuation> continuationStack;
    ArrayDeque<Object> valueStack;
    private HashMap<String,Object> environment;
    private Random rng;
    private float logW;

    //INITALIZATION MESSAGES
    public Machine(ArrayDeque<Continuation> continuationStack, ArrayDeque<Object> valueStack, HashMap<String,Object> environment, Random rng, float logW){
        this.continuationStack = continuationStack;
        this.valueStack = valueStack;
        this.environment = environment;
        this.rng = rng;
        this.logW = logW;
        
    }

    private static Machine instantiate_machine(Continuation mainEv, HashMap<String,Object> environment, Random rng){
        ArrayDeque<Continuation> continuationStack = new ArrayDeque<Continuation>();
        continuationStack.add(mainEv);

        ArrayDeque<Object> valueStack = new ArrayDeque<>();

        return new Machine(continuationStack, valueStack, environment, rng, 0.0f);
    }

    public static Machine initial_machine(String program, Random rng){
        HashMap<String,Object> genv = new HashMap<String,Object>();
        Form main = null;

        ArrayList<Form> parsedProgram = Parser.parse(program);

        for(int i = 0; i < parsedProgram.size(); i++){
            Form form = parsedProgram.get(i);

            if(form instanceof FormList formList && formList.size() != 0
            && formList.get(0) instanceof FormSymbol formSymbol
            && formSymbol.text().equals("defn")){

                String name = ((FormSymbol) formList.get(1)).text();
                ArrayList<Form> params = ((FormList) formList.get(2)).elements();
                ArrayList<Form> body = new ArrayList<>(formList.elements().subList(3, formList.size()));

                genv.put(name, new Closure(params, body, genv));
            }
            else{
                main = form;
            }
        }

        ArrayList emptyAddresses = new ArrayList();
        Ev mainEv = new Ev(main, genv, emptyAddresses);

        return Machine.instantiate_machine(mainEv, genv, rng);
    }

    public Machine fork(Random seed){

        ArrayDeque<Continuation> continuationStackCopy = new ArrayDeque<Continuation>(this.continuationStack);
        ArrayDeque<Object> valueStackCopy = new ArrayDeque<Object>(this.valueStack);
        HashMap environmentCopy = new HashMap<>(this.environment);

        return new Machine(continuationStackCopy, valueStackCopy, environmentCopy, seed, logW);
    }


    //GETTERS
    public Object getNextValue(){
        return valueStack.pop();
    }

    public Float getLogW(){
        return this.logW;
    }

    public Random getRNG(){
        return this.rng;
    }



    //"SETTERS" (Kind of, also includes updaters)

    public void increaseLogW(float value){
        this.logW += value;
    }

    public void pushValue(Object value){
        valueStack.push(value);
    }

    public void pushContinuation(Continuation continuation){
        continuationStack.push(continuation);
    }

    public void push_body(ArrayList<Form> body, HashMap environment, ArrayList address){
        ArrayList<Continuation> seq = new ArrayList<>();

        Ev ev;
        Discard discard = new Discard();
        for(int i = 0; i < body.size() - 1; i++){
            ArrayList newAddress = new ArrayList(address);
            newAddress.add("body");
            newAddress.add(i);
            ev = new Ev(body.get(i), environment, newAddress);
            seq.add(ev);
            seq.add(discard);
        }

        ArrayList newAddress = new ArrayList(address);
        newAddress.add("body");
        newAddress.add(body.size() - 1);
        ev = new Ev(body.get(body.size() - 1), environment, newAddress);
        seq.add(ev);

        for(int i = seq.size() - 1; i >= 0; i--){
            continuationStack.push(seq.get(i));
        }
    }





    public Message resume(){
        Message message;

        while (! continuationStack.isEmpty()) {
        
            Continuation nextContinuation = continuationStack.pop();

            message = nextContinuation.executeOn(this);
        
            if(message instanceof ContinueMessage){
                continue;
            }
            else{
                return message;
            }
        
            
        
        }
        message = new DoneMessage(valueStack.peek(), this);
        return message;
    }


    




}