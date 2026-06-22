#include <iostream>
#include <tuple>
#include <vector>
#include <bits/stdc++.h>
using namespace std; 


bool sortdesc(const tuple<int, int> a, 
              const tuple<int, int> b) 
{ 
    return (get<1>(a) < get<1>(b)); 
} 





int func(vector<tuple<int, int>> act_arr){
    int counter = 1;
    sort(act_arr.begin(), act_arr.end(), sortdesc);
    tuple<int,int> last_act = act_arr[0];
    for(int i = 1; i < act_arr.size(); i++){
        if(get<1>(last_act) <= get<0>(act_arr[i])){
            counter += 1;
            last_act = act_arr[i];
        }
    }
    return counter;
};




int main() 
{ 
    int test_amnt; 
    vector< vector<tuple<int, int>>> test_arrays;
    int act_amnt;
    int act_start;
    int act_end;
    int final_amount;
    cin >> test_amnt;

    for(int k = 0; k < test_amnt; k++){
        cin >> act_amnt;
        vector<tuple<int, int>> act_arr;

        

        for(int i = 0; i < act_amnt; i++){
            cin >> act_start >> act_end;
            act_arr.push_back(tuple<int, int>(act_start, act_end));
        }
        test_arrays.push_back(act_arr);

    }

    for(int j = 0; j < test_amnt; j++){
        final_amount = func(test_arrays[j]);
        cout <<  final_amount << "\n" ;
    }
    

    

    return 0;
}


