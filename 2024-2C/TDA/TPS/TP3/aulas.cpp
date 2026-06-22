#include <iostream>
#include <vector>
#include <bits/stdc++.h>
using namespace std; 


int n;
vector<vector<int>> capacity;
vector<vector<int>> adj;
vector<vector<int>> current_flow;


int bfs(int s, int t, vector<int>& parent) {
    fill(parent.begin(), parent.end(), -1);
    parent[s] = -2;
    queue<pair<int, int>> q;
    q.push({s, 10000});

    while (!q.empty()) {
        int cur = q.front().first;
        int flow = q.front().second;
        q.pop();

        for (int next : adj[cur]) {
            if (parent[next] == -1 && capacity[cur][next]) {
                parent[next] = cur;
                int new_flow = min(flow, capacity[cur][next]);
                if (next == t)
                    return new_flow;
                q.push({next, new_flow});
            }
        }
    }

    return 0;
}

int maxflow(int s, int t) {
    int flow = 0;
    vector<int> parent(n);
    int new_flow;

    while (new_flow = bfs(s, t, parent)) {
        flow += new_flow;
        int cur = t;
        while (cur != s) {
            int prev = parent[cur];
            capacity[prev][cur] -= new_flow;
            capacity[cur][prev] += new_flow;
            cur = prev;
        }
    }

    return flow;
}












int main(){
    int pasillos, aulas;
    int cantidadFinalAlumnos = 0;
    int cantidadInicialAlumnos = 0;


    //input de la cant de aulas y pasillos
    cin >> aulas >> pasillos;

     
    // tenemos 3 veces el input de aulas, donde las aulas son 0 mod 3, los nodos aux son 1 mod 3 y los finales 2 mod 3
    aulas *= 3;
    capacity = vector<vector<int>>(aulas + 2, vector<int>(aulas+ 2, 0));
    current_flow = vector<vector<int>>(aulas + 2, vector<int>(aulas + 2, 0));
    adj = vector<vector<int>>(aulas + 2);
    int s = 0;
    int t = aulas + 1;
    n = aulas + 2;


    //input del Ai, armamos el S conectado a las aulas con capacidad Ai 
    for(int i = 1; i <= aulas ;i += 3){
        cin >> capacity[s][i];
        adj[s].push_back(i);
        adj[i].push_back(s);
        cantidadInicialAlumnos += capacity[s][i];
    }

    //input de Bi, armamos el t conectado a los nodos finales
    for(int i = 3; i <= aulas; i += 3){
        int alumnosFinales;
        cin >> alumnosFinales;
        cantidadFinalAlumnos += alumnosFinales;
        capacity[i][t] = alumnosFinales;
        adj[i].push_back(t);
        adj[t].push_back(i);

        //agregamos la conexion a su correspondiente 1 
        adj[i-2].push_back(i);
        adj[i].push_back(i-2);
        capacity[i-2][i] = alumnosFinales;


        //y la conexion a su correspondiente 2
        adj[i-1].push_back(i);
        adj[i].push_back(i-1);
        capacity[i-1][i] = alumnosFinales;
    }

    //pasillos
    for(int i = 0; i < pasillos; ++ i){
        int fromx,tox,from1,to1,from2,to2;
        cin >> fromx >> tox;
        from1 = fromx * 3 - 2;
        from2 = fromx  * 3 - 1;
        to1 = tox * 3 - 2;
        to2 = tox  * 3 - 1; 

        adj[from1].push_back(to2);
        adj[to1].push_back(from2);
        adj[to2].push_back(from1);
        adj[from2].push_back(to1);
        capacity[from1][to2] = capacity[s][from1];
        capacity[to1][from2] = capacity[s][to1];
    }

    vector<vector<int>> matrizOriginal = capacity;
    vector<int>AlumnosEnviados = vector<int>(aulas + 1,0);
    int ret = maxflow(s,t);
    for(int i = 1; i <= aulas ; i += 1){
        for(int j = 1; j <= aulas; j += 1){
            capacity[i][j] =  matrizOriginal[i][j] - capacity[i][j];
        } 
    }


    if(ret == cantidadInicialAlumnos && ret == cantidadFinalAlumnos ){
        cout << "YES" << "\n";
        for(int i = 1; i <= aulas; i += 3){
            for(int j = 1; j <= aulas; j += 3){
                if(i == j) cout << capacity[i][j+2] << " "; 
                else cout << capacity[i][j+1] << " ";
            }
            cout << "\n";
        }
    }
    else cout << "NO";

    return 0;
}