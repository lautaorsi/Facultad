#include <iostream>
#include <vector>
#include <bits/stdc++.h>
using namespace std; 


bool sortasc(int a, int b) { 
    return (a < b); 
} 

int calc(vector<int> arr, int low, int distance, int high, int n_cows){
    if(low >= high){
        return high;
    }
    int stall = arr[0];
    int cows_left = n_cows - 1;
    for(int i = 1; i < arr.size(); i++){
        if(stall + distance <= arr[i]){
            stall = arr[i];
            cows_left += -1;
        }
        if(cows_left == 0){
            break;
        }
    }
    

    //si no se pudo poner a todas las vacas, bajar la distancia (el high)
    if(cows_left != 0){
         return calc(arr, low, low + (distance - low) / 2, distance, n_cows);
    }
    if(low == distance){
        return low;
    }
    //si se completa el ciclo de arriba la distancia es valida, chequeamos si hay otra mas alta subiendo el low a la distancia
    if((high - distance) / 2 == 0){
        return calc(arr, distance, distance + 1, high, n_cows);
    }
    return calc(arr, distance, distance + (high - distance ) / 2, high, n_cows);
}






int main() 
{ 
    int test_amnt; 
    int n_stalls;
    vector<tuple<vector<int>,int>> test_array;
    int stall;
    int n_cows;
    cin >> test_amnt;
    for(int i = 0; i < test_amnt; i++){
        cin >> n_stalls >> n_cows;
        vector<int> stall_numbers; 
        for(int j = 0 ; j < n_stalls; j++){
            cin >> stall;
            stall_numbers.push_back(stall);
        }
        test_array.push_back(make_tuple(stall_numbers,n_cows));
    }

    for(int i = 0; i < test_amnt; i++){
        vector<int> stall_numbers = get<0>(test_array[i]);
        int ncows = get<1>(test_array[i]);
        sort(stall_numbers.begin(),stall_numbers.end(),sortasc);
        cout << calc(stall_numbers, stall_numbers[0], stall_numbers[0] + (stall_numbers[stall_numbers.size() - 1] - stall_numbers[0])/2  , stall_numbers[stall_numbers.size() - 1], ncows) << "\n";
    }

    
  
    return 0; 
}