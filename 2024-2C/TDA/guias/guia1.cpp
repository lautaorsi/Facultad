#include <iostream>
#include <tuple>
#include <vector>
#include <bits/stdc++.h>
using namespace std; 


int ej10(vector<int> pesos, vector<int> resistencias,int indice_actual, vector<int> indices_usados){
    bool cumple_resistencia = true;
    if(indice_actual == resistencias.size()){
        return 0;
    }
    for(int i = 0; i < indices_usados.size(); i++){
        int resistencia = resistencias[indices_usados[i]];
        int peso = 0;
        for(int j = 0; j <= i; j++){
            peso = peso + pesos[indices_usados[j]];
        }
        if(resistencia < peso + pesos[indice_actual]){
            cumple_resistencia = false;
            break;
        }

    }
    if(cumple_resistencia){
        vector<int> indices_actualizados = indices_usados;
        indices_actualizados.push_back(indice_actual);
        return max(ej10(pesos, resistencias, indice_actual + 1, indices_actualizados) + 1, ej10(pesos, resistencias, indice_actual + 1, indices_usados));
    }
    else{
        return ej10(pesos, resistencias, indice_actual + 1, indices_usados);
    }
    return 0;
}


int closest_index(vector<int> seq, int target){
    for(int i = 0; i < seq.size(); i++){
        if(i == seq.size() - 1){
            return i;
        }

        if(seq[i] <= target && seq[i+1] > target){
            return i;
        }
    }
}



tuple<vector<int>,int> ej16(vector<int> seq, int c, int m){
    int stops_counter = 0;
    vector<int> stops;
    int current_position = 0;
    int next_position = 0;
    while(m > 0){
        int stop = closest_index(seq, current_position + c);
        next_position = seq[stop];
        m -= next_position - current_position;
        if(m <= 0){
            break;
        }
        current_position = next_position;
        stops.push_back(current_position);
        stops_counter++;
    }
    return make_tuple(stops, stops_counter);
}


int ej18(vector<int> seq){
    vector<int> subseq;
    for(int i = 0; i < seq.size(); i++){
        int minimo = *min_element(seq.begin(), seq.end());
        subseq.push_back(minimo);
        //retornar el min del complemento de subseq
        return 0 ;
    }

}


int maxhueco(vector<int> s){
    vector<int> fsthalf, sndhalf;
    if(s.size() != 2){
        fsthalf.assign(s.begin(), s.begin() + s.size() / 2);
        sndhalf.assign(s.begin() + s.size() / 2, s.end());
        return max(maxhueco(fsthalf), maxhueco(sndhalf));
    }
    if(s.size() == 2){
        return s[0] - s[1];
    }
}


vector<vector<int>> memo;

// int cc(int i,int j){
//    if(j <= 0){
//     return j;
//    }

//    if(j > 0 && i == - 1){
//     return -10000;
//    }



//    if(memo[i][j] != -10000){
//     return memo[i][j];
//    }
//    int res1 = cc(i-1,j-seq[i]);
//    int res2 = cc(i-1,j);
//    memo[i][j] = max(res1,res2);
//    return memo[i][j];
// }

vector<int> peso = {19,7,5,6,1};
vector<int> soporte = {15,13,7,8,2};


int pilacauta(int i, int mp){
    if(mp < 0){
        return -1000;
    }
    if(mp == 0 || i == peso.size()){
        return 0;
    }
    if(memo[i][mp] != -1){
        return memo[i][mp];
    }
    memo[i][mp] = max(pilacauta(i+1,min(mp-peso[i],soporte[i])) + 1,pilacauta(i+1,mp));
    return memo[i][mp];
}



//n, h y r da
int rec(int i, int j, int puntos){
    if(i == n && ){
        return puntos;
    }
    if()


    return puntos;
}






int main(){
    int mp = 15;
    memo = vector<vector<int>>(peso.size(), vector<int>(16, -1));
    cout << pilacauta(0,16);
    return 0;
}