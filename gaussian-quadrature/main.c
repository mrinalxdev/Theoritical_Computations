#include <stdio.h>
#include <math.h>
#include "gaussian_quadrature.h"

double testfunc(double x)
{
    return x * x;
}

int main()
{
    double a = 0.0;
    double b = 1.0;
    int n = 5;

    double result = gaussian_quadrature(testfunc, a, b, n);
    printf("Integral of x^2 from %f to %f = %.10f\n", a, b, result);
    printf("Exact value = %.10f\n", (b * b * b - a * a * a) / 3.0);
    return 0;
}
