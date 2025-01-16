#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "gaussian_quadrature.h"

static const double points[] = {
    -0.906179845938664,
    -0.538469310105683,
    0.0,
    0.538469310105683,
    0.906179845938664};

static const double weights[] = {
    0.236926885056189,
    0.478628670499366,
    0.568888888888889,
    0.478628670499366,
    0.236926885056189};


double gaus_quad(double (*f)(double), double a, double b, int n){
    if (n != 5){
        printf("Warning : This implementation only supports n=5\n");
        n=5;
    }

    double mid = (b + a)/2;
    double len = (b - a)/2;

    double result = 0.0;
    for (int i = 0; i < n; i++){
        double x = mid + len * points[i];
        result += weights[i] * f(x);
    }

    return len * result;
}
