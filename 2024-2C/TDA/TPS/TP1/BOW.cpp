#include <iostream>
#include <tuple>
#include <vector>
#include <bits/stdc++.h>
using namespace std; 





int f(const vector<int> sequence) {
    vector<vector<vector<int>>> matrix(sequence.size() + 1, vector<vector<int>>(sequence.size() + 1, vector<int>(sequence.size() + 1, 0)));
    for(int i = 1; i <= sequence.size(); i++){
        for(int j = 0; j < i; j++){
            for(int k = 0; k < i; k++){
                //"bottomupeamos" construyendo la matriz desde 0 
                    int ant = matrix[i-1][j][k];
                    matrix[i][j][k] = ant;
                    if(j == 0){
                        matrix[i][i][k]= max(matrix[i][i][k], ant + 1);
                    }

                    if(k == 0){
                        matrix[i][j][i] = max(ant + 1, matrix[i][j][i]);
                    }

                    //j seria el ascendente
                    if(j != 0 && sequence[i-1] > sequence[j-1]){
                        matrix[i][i][k]= max(matrix[i][i][k], ant + 1);
                    }

                 

                    //k el descendente
                    if(k != 0 && sequence[i-1] < sequence[k-1]){
                        matrix[i][j][i] = max(ant + 1, matrix[i][j][i]);
                    }
            }   
        }   
    }
    //el unico importante es el ult por bottom up
    vector<vector<int>> final_options = matrix[sequence.size()];
    int res = final_options[0][0];
    for(int i = 0; i <= sequence.size(); i++){
        for(int j = 0; j<= sequence.size(); j++){
            res = max(res, final_options[i][j]);
        }
    }
    return res;
}



int main(){
    int number, list_size;
    vector<vector<int>> lists_list;
    vector<int> ans_list;
    while(true){
        cin >> list_size;
        if(list_size == -1){
            break;
        }
        vector<int> number_list; 
        for(int j = 0; j < list_size; j++){
            cin >> number;
            number_list.push_back(number);
        }
        lists_list.push_back(number_list);
    }

    for(vector<int> list : lists_list){
        ans_list.push_back(list.size() - f(list));
    }
    for(int ans : ans_list){
        cout <<  ans << "\n";
    }


    return 0;
}


