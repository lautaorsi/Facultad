    #include <iostream>
    #include <vector>
    #include <bits/stdc++.h>
    using namespace std; 
    class DisjointSet {
        vector<int> rank, parent;
    public:
        DisjointSet(int n) {
            rank.resize(n + 1, 0);
            parent.resize(n + 1);
            for(int i = 0; i <= n; i++){
                parent[i] = i;
            }
        }
    
        int findSet(int node){
    
            // En caso que nodo sea el representante
            if (node == parent[node]) return node;
    
            // Hago path compression
            return parent[node] = findSet(parent[node]);
        }
    
        void unionByRank(int u, int v) {
            int uRepresentative = findSet(u);
            int vRepresentative = findSet(v);
    
            // Si tienen el mismo representante, entonces pertenece al
            // mismo conjunto
            if (uRepresentative == vRepresentative) return;
    
            // Actualizamos el representante segun el caso del rank
            if (rank[uRepresentative] < rank[vRepresentative]) {
                parent[uRepresentative] = vRepresentative;
            } else if(rank[uRepresentative] > rank[vRepresentative]) {
                parent[vRepresentative] = uRepresentative;
            } else {
                parent[vRepresentative] = uRepresentative;
                rank[uRepresentative]++;
            }
        }
    };
    



    int parent;
    unordered_map<int, unordered_map<int,string>> output;
    map<int, unordered_map<int,vector<int>>> matrizPorPesos;
    unordered_map<int,vector<int>> matrizAdy;
    unordered_map<int,int> cant_peso;
    int NO_LO_VI = 0, EMPECE_A_VER = 1, TERMINE_DE_VER = 2;
    vector<int> estado;
    vector<int> memo;
    vector<int> padre;
    vector<vector<int>> tree_edges;
    vector<int> back_edges_con_extremo_inferior_en;
    vector<int> back_edges_con_extremo_superior_en;
    unordered_map<int, bool> raices;

    void dfs(unordered_map<int,vector<int>>& matriz, int v, int p = -1) {
        estado[v] = EMPECE_A_VER;
        for (int u : matriz[v]) {
            if (estado[u] == NO_LO_VI) {
                tree_edges[v].push_back(u);
                padre[u]=v;
                dfs(matriz, u, v);
            }
            else if (u != padre[v]) {
                if (estado[u] == EMPECE_A_VER) {
                    back_edges_con_extremo_superior_en[v]++;
                }
                else // estado[u] == TERMINE_DE_VER
                    back_edges_con_extremo_inferior_en[v]++;
            }
        }
        estado[v] = TERMINE_DE_VER;
    }

    int cubren(int v, int p) {
        if (memo[v] != -1) return memo[v];
        int res = 0;
        for (int hijo : tree_edges[v]) {
            if (hijo != p) {
                res += cubren(hijo, v);
            }
        }
        res += back_edges_con_extremo_superior_en[v];
        res -= back_edges_con_extremo_inferior_en[v];
        memo[v] = res;
        return res;
    }



    
    //codigo de kruskal sacado de la practica (modificado evidentemente)
    void kruskal(vector<tuple<int,int,int>>& E, int n){
        estado = vector<int>(n+1,0);
        memo = vector<int>(n+1,-1);
        padre = vector<int>(n+1,-1);
        tree_edges = vector<vector<int>>(n+1);
        back_edges_con_extremo_inferior_en = vector<int>(n+1,0);
        back_edges_con_extremo_superior_en = vector<int>(n+1,0);

        int cant_nodos_agregados = 0;
        unordered_map<int, bool> peso_recorrido;
        sort(E.begin(),E.end());
        DisjointSet dsu = DisjointSet(n); 
        for(auto& aristasDePeso : matrizPorPesos){ //O(cant_pesos)

            int peso = get<0>(aristasDePeso);
            auto aristas = get<1>(aristasDePeso);

            //si hay mas de una arista con ese peso
            if(cant_peso[peso] > 1){
                
                unordered_map<int, unordered_map<int, bool>> ignorar_arista;
                //armo una copia 
                unordered_map<int,vector<int>> copia_matriz = matrizAdy;
                
                //si alguna arista forma ciclo sola es none O(E)
                int cant_agregadas = 0;
                for(auto& [u,vs] : aristas){
                    for(int i =0; i < vs.size(); i++){
                        int v = vs[i];
                        if(dsu.findSet(u) == dsu.findSet(v)){
                            output[u][v] = "none";
                            ignorar_arista[u][v] = true;
                        }
                    }
                }




                //agrego todas las aristas a la copia O(E)
                for(auto& [u,vs] : aristas){
                    for(int i =0; i < vs.size(); i++){
                        int v = vs[i];
                        if(!ignorar_arista[u][v]){
                            cant_agregadas += 1;
                            cant_nodos_agregados += 1;
                            copia_matriz[u].push_back(v);
                            copia_matriz[v].push_back(u);
                        }

                    }
                }
                if(cant_agregadas == 0){
                    continue;
                }

                
                //limpiamos O(V + E)
                tree_edges = vector<vector<int>>(n+1);
                for(auto& [u,vs] : matrizAdy){
                    memo[u] = -1;
                    back_edges_con_extremo_inferior_en[u] = 0;
                    back_edges_con_extremo_superior_en[u] = 0;
                    padre[u] = -1;
                    estado[u] = 0;
                    for(int v : vs){
                        memo[v] = -1;
                        back_edges_con_extremo_inferior_en[v] = 0;
                        back_edges_con_extremo_superior_en[v] = 0;
                        padre[v] = -1;
                        estado[v] = 0;
                    }
                }

                //para cada cc hago dfs y me guardo la raiz O(V + E)
                for(auto& [u, vs] : copia_matriz){
                    if(estado[u] == 0){
                        raices[u] = true;
                        dfs(copia_matriz, u);
                    }
                }

                //O(V + E)
                for(auto& [raiz, esraiz] : raices){
                    cubren(raiz, -1);
                }
                
                
                //busco las puentes O(E)
                for(auto& [u,vs] : aristas){
                    for(int i =0; i < vs.size(); i++){
                        int v = vs[i];
                        if(ignorar_arista[u][v]) continue;
                        //si no es raiz y no esta cubierta es any
                        if(padre[v] == u && memo[v] == 0){
                            output[u][v] = "any";
                        }
                        else if(padre[u] == v && memo[u] == 0) {
                            output[u][v] = "any";
                        }
                        //si no es puente esta en un ciclo
                        else{
                            output[u][v] = "at least one";
                        }
                        matrizAdy[u].push_back(v);
                        matrizAdy[v].push_back(u);
                        dsu.unionByRank(u,v);
                    }
                }

                


            }
            //si hay una sola con ese peso
            else{
                for(auto& [u,vs] : aristas){
                    for(int i =0; i < vs.size(); i++){
                        int v = vs[i];
                        //si no forma ciclo es any
                        if(dsu.findSet(u) != dsu.findSet(v)){
                            output[u][v] = "any";
                            dsu.unionByRank(u,v);
                            matrizAdy[u].push_back(v);
                            matrizAdy[v].push_back(u);
                            cant_nodos_agregados += 1;
                        }
                        //si forma ciclo no la agrego
                        else{
                            output[u][v] = "none";
                        }
                    }

                }

            }


            }
        }

    





    
    int main() {
        int cantNodos, cantVertices;
        cin >> cantNodos >> cantVertices;
        vector<tuple<int,int,int>>E;
        vector<tuple<int, int>> aristas;
    

        for(int i = 0; i < cantVertices; i++) {
            int fila, columna, peso;
            cin >> fila >> columna >> peso;
            E.push_back(make_tuple(peso, fila, columna));
            matrizPorPesos[peso][fila].push_back(columna);
            cant_peso[peso] += 1;
            aristas.push_back(make_tuple(fila, columna));
        }
    
        kruskal(E, cantNodos);
        for(auto& [u,v] : aristas){
            cout << output[u][v] << "\n";
        }

    }




