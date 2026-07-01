import java.util.Dictionary;
import java.util.List;

public class Interface{




    public Tuple<String, Int, Machine> resume(Machine machine){

        while(machine.hasInstructions()){
            machine.executeNextInstruction();
        }
        
        Tuple<String, Int, Machine> return_value = new Tuple("done", V[-1], m);

        return return_value;
    }



    public void send(Machine machine, T Value){
        machine.pushValue(value);
    }


    public void initial_machine(String program, Int seed){
        new_env = new Dictionary<K,valueStack>();
        main = null;
        parsed_program = parse(program);
        for(int i = 0; i < parsed_program.length(); i++){
            form = parsed_program.get(i);
            if(form instanceof List && form != null && form.get(0) == "defn"){
                name = form.get(2);
                params = form.get(3);
                body = form.get(4);
                new_env.put(name, Closure(params, body, new_env));
            }
            else{
                main = form;
            }
        }

        Expression mainExpression = new Expression.createExpression(main);

        Machine machine = Machine.instantiate_initial_machine(seed, new_env, mainExpression);
    }
    
}