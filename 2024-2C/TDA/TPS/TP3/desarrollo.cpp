#include <iostream>
#include <vector>
#include <bits/stdc++.h>
using namespace std; 


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
            res += matriz[j][i];
        }
    }
    return res;
}






vector<vector<int>> sacar(vector<vector<int>>& matriz, int nodo){
    vector<vector<int>> nuevaMatriz = vector<vector<int>>(matriz.size() - 1, vector<int>(matriz.size() - 1));
    for(int i = 0; i < matriz.size(); i++){
        for(int j = 0; j < matriz.size(); j++){
            if(i != nodo && j != nodo){
                int nuevoi;
                int nuevoj;
                if(i < nodo){
                    nuevoi =i;
                }
                else{
                    nuevoi = i -1;
                }
                if(j < nodo){
                    nuevoj = j;
                }
                else{
                    nuevoj = j-1;
                }
                nuevaMatriz[nuevoi][nuevoj] = matriz[i][j];
            }
        }
    }
    return nuevaMatriz;
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

    int i = 0;
    while(cantNodos != 1){
        vector<vector<int>> nuevaMatriz = dantzig_sin_negativos(matriz);
        cout << sumatoria_matriz(nuevaMatriz, cantNodos) << " ";
        matriz = sacar(matriz,orden[i] + i - 1);
        cantNodos -= 1;
        i++;
    }
    cout << 0;
    return 0;
}