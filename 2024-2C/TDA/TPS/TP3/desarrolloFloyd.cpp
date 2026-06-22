#include <iostream>
#include <vector>
#include <bits/stdc++.h>
using namespace std; 

//retornar la sumatoria de distnacias todos a todos, sacando nodos en O(n^3)






long long sumatoria_matriz(vector<vector<long long>>& matriz, vector<bool> usados){ //O(N^2)
    long long res = 0;
    for(long long j = 0 ; j < matriz.size(); j++){
        if(!(usados[j])) continue;
        for(long long i = 0 ; i < matriz.size(); i++){
            if(!(usados[i])) continue;
            res += matriz[j][i];
        }
    }
    return res;
}



vector<long long> InvarianteFloyd(vector<long long> orden,vector<vector<long long>> actual){         //O(N^3)  
    long long n = actual.size();
    vector<long long> res;
    vector<bool> usados = vector<bool>(n,false);
    long long k;
    for (long long indice = n-1; indice >= 0 ; indice--) {
        k = orden[indice];
        usados[k] = true;
        for (long long i = 0; i < n; ++i) {
            if(!usados[i]) continue;
            for (long long j = 0; j < n; ++j) {
                if(!usados[j])continue;
                actual[k][i] = min(actual[k][i], actual[k][j] + actual[j][i]);
                actual[i][k] = min(actual[i][k], actual[i][j] + actual[j][k]); 

            }
        }

        for (long long i = 0; i < n; ++i) {
            if(!usados[i]) continue;
            for (long long j = 0; j < n; ++j) {
                if(!usados[j])continue;
                actual[i][j] = min(actual[i][j], actual[i][k] + actual[k][j]); 
                actual[j][i] = min(actual[j][i], actual[j][k] + actual[k][i]); 
            }
        }









        res.push_back(sumatoria_matriz(actual,usados));
    }
    return res;
}



int main(){
    long long cantNodos, peso;

    //cantidad de nodos
    cin >> cantNodos;

    //inicializo matriz, vector de retorno y matriz sinK
    vector<vector<long long>> matriz = vector<vector<long long>>(cantNodos, vector<long long>(cantNodos, 100001));



    //guardo los pesos en la matriz
    for(long long i = 0 ; i <= cantNodos - 1; i++){
        for(long long j = 0; j <= cantNodos - 1; j++){
            cin >> peso;
            matriz[i][j] = peso;
        }
    }

    //recibo orden de eliminacion
    vector<long long> orden;
    for(long long i = 0; i < cantNodos; i++){
        long long numero;
        cin >> numero;
        orden.push_back(numero - 1);
    }


    vector<long long> res = InvarianteFloyd(orden,matriz);

    for(long long i = res.size() -1; i >= 0; i--){
        cout << res[i] << " ";
    }
    return 0;
}