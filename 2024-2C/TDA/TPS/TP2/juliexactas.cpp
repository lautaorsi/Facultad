#include <bits/stdc++.h>
using namespace std;

vector<int> retornos;
int V;

void primMST(unordered_map<int, unordered_map<int, int>>& diccpeso) {
    vector<int> parent(V + 1, -1);
    vector<int> energiaMinima(V + 1, INT_MAX);
    vector<bool> mstSet(V + 1, false);
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;

    energiaMinima[1] = 0;
    pq.push({0, 1}); 

    while (!pq.empty()) {
        int u = pq.top().second;
        pq.pop();
        
        if (!mstSet[u]){
            mstSet[u] = true;
            for(auto adyacente : diccpeso[u]){
                int i = adyacente.first;
                //actualizamos la energia de las aristas adyacentes al de la arista q conectamos y los reemplazamos en energiaMinima si son menores
                if(parent[u] != -1 && !mstSet[i] && diccpeso[u][i] + diccpeso[parent[u]][u] < energiaMinima[i]){
                    diccpeso[u][i] += diccpeso[parent[u]][u]; 
                    energiaMinima[i] = diccpeso[u][i];
                    parent[i] = u;
                }
                if(parent[u] == -1 && !mstSet[i] && diccpeso[u][i] < energiaMinima[i]){
                    energiaMinima[i] = diccpeso[u][i];
                    parent[i] = u;
                }
                pq.push({energiaMinima[i], i});
            }
        }

    }

    for (int i = 1; i <= V; i++) {
        if (parent[i] != -1) {
            retornos.push_back(diccpeso[parent[i]][i]);
        }
    }
}
int main(){
    int cantidad_aulas;
    cin >> cantidad_aulas;
    V = cantidad_aulas;
    vector<int> tuneles = vector<int>(cantidad_aulas +1, 0);
    for(int i = 1; i <= cantidad_aulas; i++){
        cin >> tuneles[i];
    }
    unordered_map<int, unordered_map<int, int>> DiccPesos;
    //agrego los puentes cambiando en la matriz de ady el peso j-i a 1
    for(int k = 1; k <= cantidad_aulas; k++){
        if(k != tuneles[k]){
            DiccPesos[k][tuneles[k]] = 1;
        }
        if(k != 1){
            DiccPesos[k][k-1] = 1; 
        }
        if(k != cantidad_aulas){
            DiccPesos[k][k+1] = 1;
        }
    }



 
    //hacemos un prim pero vamos actualizando dinamicamente el peso de las aristas sumandole el peso de la ult arista que recorrimos
    primMST(DiccPesos);
    cout << 0 << " ";
    for(int elem : retornos){
        cout << elem << " ";
    }
    return 0;
}