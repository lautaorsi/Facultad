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
    unordered_map<int, unordered_map<int, int>> cant_agregada;

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
                cant_agregada.clear();
                ignorar_arista.clear();

                //armo una copia 
                unordered_map<int,vector<int>> copia_matriz = matrizAdy;
                unordered_map<int,unordered_map<int, tuple<int, int>>> primeros;
                //si alguna arista forma ciclo sola es none O(E)
                for(auto& [u,vs] : aristas){
                    for(int i =0; i < vs.size(); i++){
                        int v = vs[i];
                        if(dsu.findSet(u) == dsu.findSet(v)){
                            output[u][v] = "none";
                            ignorar_arista[u][v] = true;
                        }
                    }
                }




                //agrego los representantes a la copia
                for(auto& [u,vs] : aristas){
                    for(int i =0; i < vs.size(); i++){
                        int v = vs[i];
                        if(!ignorar_arista[u][v]){
                            if(cant_agregada[dsu.findSet(u)][dsu.findSet(v)] == 0){
                                copia_matriz[dsu.findSet(u)].push_back(dsu.findSet(v));
                                copia_matriz[dsu.findSet(v)].push_back(dsu.findSet(u));
                                cant_agregada[dsu.findSet(u)][dsu.findSet(v)] = 1;
                                primeros[dsu.findSet(u)][dsu.findSet(v)] = make_tuple(u,v);
                                continue;    
                            }
                            if(cant_agregada[dsu.findSet(u)][dsu.findSet(v)] != 0){
                                cant_agregada[dsu.findSet(u)][dsu.findSet(v)] = 2;
                                output[u][v] = "at least one";
                                output[get<0>(primeros[dsu.findSet(u)][dsu.findSet(v)])][get<1>(primeros[dsu.findSet(u)][dsu.findSet(v)])] = "at least one";
                                output[dsu.findSet(u)][dsu.findSet(v)] = "at least one";
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


                for(auto& [raiz, esraiz] : raices){
                    cubren(raiz, -1);
                }
                
                
                //busco las puentes O(E)
                for(auto& [u,vs] : copia_matriz){
                    for(int i =0; i < vs.size(); i++){
                        int v = vs[i];
                        if(ignorar_arista[u][v]) continue;
                        //si no es raiz y no esta cubierta es any
                        if(cant_agregada[u][v] != 2 && padre[v] == dsu.findSet(u) && memo[(v)] == 0){
                            output[u][v] = "any";
                            matrizAdy[u].push_back(v);
                            matrizAdy[v].push_back(u);
                        }
                        else if(cant_agregada[u][v] != 2 && padre[u] == dsu.findSet(v) && memo[(u)] == 0) {
                            output[u][v] = "any";
                            matrizAdy[v].push_back(u);
                            matrizAdy[u].push_back(v);
                        }
                        else{
                            output[u][v] = "at least one";
                        }
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




