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
    unordered_map<int, int> estado;
    unordered_map<int, int> memo;
    unordered_map<int, int> padre;
    unordered_map<int, vector<int>> tree_edges;
    unordered_map<int, int> back_edges_con_extremo_inferior_en;
    unordered_map<int, int> back_edges_con_extremo_superior_en;
    unordered_map<int, bool> raices;
    unordered_map<int, unordered_map<int, bool>> ignorar_arista;
    unordered_map<int, unordered_map<int, bool>> ya_agregada;

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
        if (memo.find(v) != memo.end()) return memo[v];
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
        unordered_map<int, bool> peso_recorrido;
        sort(E.begin(),E.end());
        DisjointSet dsu = DisjointSet(n); 
        for(auto& aristasDePeso : matrizPorPesos){ //O(cant_pesos)

            int peso = get<0>(aristasDePeso);
            auto aristas = get<1>(aristasDePeso);

            //si hay mas de una arista con ese peso
            if(cant_peso[peso] > 1){
                



                estado.clear();
                memo.clear();
                padre.clear();
                tree_edges.clear();
                back_edges_con_extremo_inferior_en.clear();
                back_edges_con_extremo_superior_en.clear();
                raices.clear();
                ignorar_arista.clear();
                
                //armo una copia 
                unordered_map<int,vector<int>> copia_matriz;
                unordered_map<int,unordered_map<int, vector<tuple<int,int>>>> originales;
                unordered_map<int, unordered_map<int, bool>> aristas_agregadas;
                vector<tuple<int,int>> leastear;
                //si alguna arista forma ciclo sola es none O(E)
                for(auto& [u,vs] : aristas){
                    for(int i =0; i < vs.size(); i++){
                        int v = vs[i];
                        if(dsu.findSet(u) == dsu.findSet(v)){
                            output[u][v] = "none";
                            output[v][u] = "none";
                            ignorar_arista[u][v] = true;
                            ignorar_arista[v][u] = true;
                        }
                    }
                }




                //agrego las aristas, usando representantes (que son ellos mismos si el nodo no esta en el AGM) <- G'
                for(auto& [u,vs] : aristas){
                    for(int i =0; i < vs.size(); i++){
                        int v = vs[i];
                        if(!ignorar_arista[u][v]){
                            
                            //agregamos la arista original a la lista de las aristas dentro de la representante
                            originales[dsu.findSet(u)][dsu.findSet(v)].push_back(make_tuple(u,v));
                            
                            if(!aristas_agregadas[dsu.findSet(u)][dsu.findSet(v)] && !aristas_agregadas[dsu.findSet(v)][dsu.findSet(u)]){
                                //agregamos la arista entre representantes
                                copia_matriz[dsu.findSet(u)].push_back(dsu.findSet(v));
                                copia_matriz[dsu.findSet(v)].push_back(dsu.findSet(u));
                                aristas_agregadas[dsu.findSet(u)][dsu.findSet(v)] = true;
                            }
                            else{
                                leastear.push_back(make_tuple(dsu.findSet(u), dsu.findSet(v)));
                                leastear.push_back(make_tuple(dsu.findSet(v), dsu.findSet(u)));
                            }
                        }
                    }
                }


                


                //para cada cc hago dfs y me guardo la raiz
                for(auto& [u, vs] : copia_matriz){
                    if(estado[u] == 0){
                        raices[u] = true;
                        dfs(copia_matriz, u);
                    }
                }

                //hago cubren para cada cc
                for(auto& [raiz, esraiz] : raices){
                    cubren(raiz, -1);
                }
                
                
                //busco las puentes de las aristas, G es un grafo normal
                for(auto& [u,vs] : copia_matriz){
                    for(int i =0; i < vs.size(); i++){
                        int v = vs[i];
                            if((padre[u] == v && memo[u] == 0) || (padre[v] == u && memo[v] == 0)){
                                output[u][v] = "any";
                            }
                            else{
                                output[u][v] = "at least one";
                            }
                    }
                }

                //ponemos a las aristas originales el valor de las representantes
                for(auto& [u,vs] : copia_matriz){
                    for(int i =0; i < vs.size(); i++){
                        int v = vs[i];
                        for(int j = 0; j < originales[u][v].size(); j++){
                            int u1 = get<0>(originales[u][v][j]);
                            int v1 = get<1>(originales[u][v][j]);
                            if(!ignorar_arista[u1][v1]){
                               output[u1][v1] = output[u][v]; 
                            }
                            
                        }
                    }
                }


                //le ponemos at least one a los originales con representantes repetidos
                for(int i = 0 ; i < leastear.size(); i++){
                    int u = get<0>(leastear[i]);
                    int v = get<1>(leastear[i]);
                    for(int j = 0; j < originales[u][v].size(); j++){
                        int u1 = get<0>(originales[u][v][j]);
                        int v1 = get<1>(originales[u][v][j]);
                        output[u1][v1] = "at least one";
                    }
                }



                //juntamos representantes con las nuevas aristas
                for(auto& [u,vs] : copia_matriz){
                    for(int i =0; i < vs.size(); i++){
                        int v = vs[i];
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
                            output[v][u] = "any";
                            dsu.unionByRank(u,v);
                            matrizAdy[u].push_back(v);
                            matrizAdy[v].push_back(u);
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




