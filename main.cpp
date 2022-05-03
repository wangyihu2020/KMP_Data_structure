#include <iostream>
#include <string>
#include <vector>


std::vector<int> getArray(std::string& s){
    std::vector<int> aux(s.size(), 0);
    aux[0] = -1;
    aux[1] = 0;
    int cn = 0;
    for(int i = 2; i < s.size(); ){
        if(s[i - 1] == s[cn]) aux[i++] = ++cn;
        else if(cn != 0) cn = aux[cn];
        else i++;
    }
    return aux;
}
int checkifinclude(std::string& S, std::string& s){
    std::vector<int> s_array(getArray(s));
    int i = 0;
    int cn = 0;
    while(i < S.size() && cn < s.size()){
        if(s[cn] == S[i]){
            cn++;
            i++;
        }
        else if(cn != 0){
            cn = s_array[cn];
        }
        else i++;
    }
    if(cn < s.size() )
        return -1;
    else return i - cn;
}
int main() {
    std::string S("abcdefabcdefabcdfg");
    std::string s("g");
    int res = checkifinclude(S, s);


    std::cout << "Hello, World!" << std::endl;
    return 0;
}
