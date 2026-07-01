import java.util.HashMap;

public class Closure{

    private HashMap env;
    private ArrayList params, body;


    public Closure(ArrayList params, ArrayList body, HashMap env){
        self.params = params;
        self.body = body;
        self.env = env; 
    }


}

