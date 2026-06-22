#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

vector<vector<vector<int>>> matrix;



int recursion(vector<int>& sequence, int c_i, int ic, int id){
    if(c_i == sequence.size()){
        return sequence.size();
    }

    if(matrix[c_i][ic + 1][id + 1] != -1){
        return matrix[c_i][ic + 1][id + 1];
    }

    int ans = recursion(sequence, c_i + 1, ic, id);
    if(ic == -1  || sequence[c_i] > sequence[ic]){
       ans = min(recursion(sequence, c_i + 1, c_i, id) - 1,  ans);
    }

    if(id == -1 || sequence[c_i] < sequence[id]){
       ans =  min(recursion(sequence, c_i + 1, ic, c_i)  - 1, ans);
    }

    matrix[c_i][ic + 1][id + 1] = ans;
    return ans;
}





int f(vector<int> sequence){
    matrix = vector<vector<vector<int>>>(sequence.size() + 1, vector<vector<int>>(sequence.size() + 1, vector<int>(sequence.size() + 1, -1)));
    return recursion(sequence, 0,-1,-1) ;
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
        cout<< f(list) << "\n";
    }


    return 0;
}