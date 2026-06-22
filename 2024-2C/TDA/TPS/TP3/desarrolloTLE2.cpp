#include <iostream>
#include <vector>
#include <bits/stdc++.h>
using namespace std; 

//es void, paso por referencia y modifico el parametro
vector<vector<int>> dantzig_sin_negativos(vector<vector<int>> matriz){            //O(n^3)
    for(int k = 0; k < matriz.size() - 1; k++){
        for(int i = 0; i <= k; i++){
            int minimo1 = 100001;
            for(int j = 0; j <= k; j++){
                minimo1 = min(minimo1, matriz[i][j] + matriz[j][k+1]);
            }
            matriz[i][k+1] = minimo1;

            int minimo2 = 100001;
            for(int j = 0; j <= k; j++){
                minimo2 = min(minimo2, matriz[k+1][j] + matriz[j][i]);
            }
            matriz[k+1][i] = minimo2;
        }

        for(int i = 0 ; i <= k ; i++){
            for(int j = 0; j <= k; j++){
                matriz[i][j] = min(matriz[i][j], matriz[i][k+1] + matriz[k+1][j]);
            }
        }
    }
    return matriz;
}


int sumatoria_matriz(vector<vector<int>>& matriz, int cantNodos){
    int res = 0;
    for(int j = 0 ; j < cantNodos; j++){
        for(int i = 0 ; i < cantNodos; i++){
            if(matriz[j][i] != 100001) res += matriz[j][i];
        }
    }
    return res;
}





void sacar(vector<vector<int>>& matriz, int nodo){
    for(int i = 0; i < matriz.size(); i++){
        matriz[nodo][i] = 100001;
        matriz[i][nodo] = 100001;
    }
}


int main(){
    //recibimos matriz de pesos
    int cantNodos, peso;
    cin >> cantNodos;
    vector<vector<int>> matriz = vector<vector<int>>(cantNodos, vector<int>(cantNodos));
    for(int i = 0 ; i <= cantNodos - 1; i++){
        for(int j = 0; j <= cantNodos - 1; j++){
            cin >> peso;
            matriz[i][j] = peso;
        }
    }

    vector<int> orden;
    for(int i = 0; i < cantNodos; i++){
        int numero;
        cin >> numero;
        orden.push_back(numero);
    }

    for(int i = 0; i < cantNodos; i++){
        vector<vector<int>> nuevaMatriz = dantzig_sin_negativos(matriz);
        cout << sumatoria_matriz(nuevaMatriz, cantNodos) << " ";
        sacar(matriz,orden[i] - 1);
    }
    return 0;
}