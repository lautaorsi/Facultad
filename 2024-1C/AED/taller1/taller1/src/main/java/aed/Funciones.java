package aed;

class Funciones {
    int cuadrado(int x) {
        return (x*x);
    }

    double distancia(double x, double y) {
        return Math.sqrt(x*x + y*y);
    }



    boolean esPar(int n) {
        //if n mod 2 = 0 es par
        if((n % 2 )== 0){

            return true;
        }
        return false;
    }

    boolean esBisiesto(int n) {
        //if n div 4 y n NOdiv 100 ====> true
        //O n div 400 ===> true
        if(((n % 4) == 0 && (n % 100) != 0) || (n % 400 == 0)){
            return true;
        }
        return false;
    }

    int factorialIterativo(int n) {
        //si n es 0 devolver 1
        if(n == 0){
            return(1);
        }

        //inicializo contador
        int i = 1;
        
        //inicializo variable de retorno
        int value = 1;

        //recorrer desde 1 hasta n y multiplicar var de retorno por el numero
        while(i <= n){
            System.out.println(n);
            System.out.println(i);
            value = value * i;
            i = i + 1;
        }
        return(value);
    }

    int factorialRecursivo(int n) {
        //retorna 1 si n = 0
        if(n == 0){
            return(1);
        }
        else{
            //recursividad, n * recurs(n)
            return(n * factorialRecursivo(n-1));
        }
    }

    boolean esPrimo(int n) {
        //si n es 0 o 1 entonces false
        if(n == 0 || n == 1){
            return false;
        }
        //inicializo contador de divisores 
        int div_counter = 0;
        //inicializo contador para iterar
        int i = 1;
        while(i <= n){
            if(n % i == 0 ){
                //si i divide, aumentar cant de div
                div_counter = div_counter + 1;
                //si tiene mas de 2 div, no es primo
                if(div_counter > 2){
                    return false;
                }
            }
            i = i + 1; 
        }
        return true;
    }

    int sumatoria(int[] numeros) {
        int suma = 0;
        for(int i = 0; i < numeros.length; i++){
            suma = suma + numeros[i];
        }
        return suma;
    }

    int busqueda(int[] numeros, int buscado){
        int res = 0;
        for(int i = 0; i < numeros.length; i++){
           if(buscado == numeros[i]){
            res = i;
           }
        }
        return(res);
    }

    boolean tienePrimo(int[] numeros) {
        for(int i = 0; i < numeros.length; i++){
            if(esPrimo(numeros[i])){
                return true;
            }
        }
        return false;
    }

    boolean todosPares(int[] numeros) {
        for(int i = 0; i < numeros.length; i++){
            if(esPar(numeros[i]) == false){
                return false;
            }
        }
        return true;
    }

    boolean esPrefijo(String s1, String s2) {
        if(s1.length() > s2.length()){
            return false;
        }
        for(int i = 0; i < s1.length(); i++){
            if(s1.charAt(i) != s2.charAt(i)){
                return false;
            }
        }
        return true;
    }



    boolean esSufijo(String s1, String s2) {
        String inv2 = "";
        String inv1 = "";
        for(int i = 0; i < s2.length(); i++){
            inv2 = s2.charAt(i) + inv2;
        }
        for(int i = 0; i < s1.length(); i++){
            inv1 = s1.charAt(i) + inv1;
        }
        return(esPrefijo(inv1, inv2));
    }






}




