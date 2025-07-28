#include <stdio.h>

int facto(int x){
    int i ;
    int fact_a = 1;
    for(i = 1; i<x; i++){
        fact_a *= i;
    }
    return fact_a;
}
int main(){

    int a = 10;
    int y = facto(a);
    printf("%d",y);
    return 0;
}