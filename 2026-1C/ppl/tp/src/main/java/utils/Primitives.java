package utils;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

import distributions.Normal;

public class Primitives {

    public interface Primitive extends Function<List<Object>, Object> {}

    private static final Map<String, Primitive> TABLE = new HashMap<>();

    // Static initializer block to populate the table
    static {
        TABLE.put("normal", args -> new Normal(
            ((Number) args.get(0)).floatValue(), 
            ((Number) args.get(1)).floatValue()
        ));
        
        // Add other primitives here...
    }

    public static boolean isPrimitive(String name) {
        return TABLE.containsKey(name);
    }

    public static Primitive getPrimitive(String name) {
        return TABLE.get(name);
    }
}