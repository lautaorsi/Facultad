
//IDEA: Djikstra y chequear distancias 

#include <iostream>
#include <vector>
#include <bits/stdc++.h>
using namespace std; 


const int INF = 1000000000;
vector<vector<pair<int, int>>> adj;


void dijkstra(int s, vector<int> & d, vector<int> & p) {
    int n = adj.size();
    d.assign(n, INF);
    p.assign(n, -1);
    vector<bool> u(n, false);

    d[s] = 0;
    for (int i = 0; i < n; i++) {
        int v = -1;
        for (int j = 0; j < n; j++) {
            if (!u[j] && (v == -1 || d[j] < d[v]))
                v = j;
        }

        if (d[v] == INF)
            break;

        u[v] = true;
        for (auto edge : adj[v]) {
            int to = edge.first;
            int len = edge.second;

            if (d[v] + len < d[to]) {
                d[to] = d[v] + len;
                p[to] = v;
            }
        }
    }
}



//arista (x,y) util <=> d(v,x) + c + d(y,w) = d(v,w)

//Ahora, como no podemo hacer d(y,w) (notese d(x,y) representa distancia desde X hasta Y) hacemos dijkstra desde 0 y n-1, luego podemos hacer 
// arista (x,y) util <=> d(v,x) + c + d(w,y) = d(v,w)  


int distancia(int nodo,vector<int> distanciaV,vector<int> distanciaW, vector<tuple<int,int, int>> aristas){
    int res = 0;

    for(auto [x,y,c] : aristas){
        if(distanciaV[x] + c + distanciaW[y] == distanciaV[nodo] || distanciaV[y] + c + distanciaW[x] == distanciaV[nodo] ){
            res += 2*c;
        }
    }

    return res;

}




int main(){

    int nodos, aristas, v, w, c;
    vector<int> d, p,d2,p2;
    
    cin >> nodos >> aristas;

    adj = vector<vector<pair<int,int>>>(nodos);
    d  = vector<int>(nodos);
    p  = vector<int>(nodos);
    d2  = vector<int>(nodos);
    p2  = vector<int>(nodos);
    vector<tuple<int,int, int>> lista_aristas;


    for(int i = 0; i < aristas; i++){
        cin >> v >> w >> c;
        adj[v].push_back(make_pair(w, c));
        adj[w].push_back(make_pair(v, c));
        lista_aristas.push_back(make_tuple(v,w, c));
    }
    
    dijkstra(0, d,  p);
    dijkstra(nodos-1, d2, p2);

    cout << distancia(nodos-1 ,d, d2, lista_aristas);


    return 0;
}