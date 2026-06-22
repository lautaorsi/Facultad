#include <iostream>
#include <vector>
#include <bits/stdc++.h>
using namespace std; 

int cant_part_1 = 0;
int cant_part_2 = 0;
int cant_nodos;
vector<bool> bi1, bi2;


void DFSRec(vector<vector<long long>> &adj, vector<bool> &visitados, long long s, bool partitud1){
    visitados[s] = true;
    for (long long i : adj[s])
        if (visitados[i] == false){
            if(partitud1){
                cant_part_1 += 1;
                bi1[i] = true;
            }
            else{
                cant_part_2 += 1;
                bi2[i] = true;
            }
            DFSRec(adj, visitados, i, !partitud1);
        }
}

void DFS(vector<vector<long long>> &adyacencias, long long s){
    vector<bool> visited(adyacencias.size(), false);
    DFSRec(adyacencias, visited, s, false);
}




long long f(vector<vector<long long>> v1, long long totales){
    long long res = 0;
    bi1 = vector<bool>(cant_nodos + 1, false);
    bi2 = vector<bool>(cant_nodos + 1, false);
    DFS(v1, 1);
    bi1[1] = true;
    cant_part_1 += 1;
    for(long long i = 0; i < v1.size(); i++){
        if(bi1[i]){
            res += ((totales - cant_part_1) - v1[i].size());
        }
    }
    return (res) ;
}




int main(){
    vector<vector<long long>> v1;
    cin >> cant_nodos;
    v1 = vector<vector<long long>>(cant_nodos + 1);
    long long nodo1, nodo2;
    for(long long i = 1 ; i < cant_nodos; i++){
        cin >> nodo1 >> nodo2;
        v1[nodo1].push_back(nodo2);
        v1[nodo2].push_back(nodo1);
    }
    cout << f(v1, cant_nodos);
    return 0;
}