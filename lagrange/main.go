package main

import (
	"errors"
	"fmt"
	"math"
)

type LagrangePolynomial struct {
	xPoints []float64
	yPoints []float64
	weights []float64
}

func NewLagrangePolynomial(xPoints, yPoints []float64) (*LagrangePolynomial, error) {
	if len(xPoints) != len(yPoints) {
		return nil, errors.New("xPoints and yPoints must have the same length")
	}
	n := len(xPoints)
	weights := make([]float64, n)

	// precomputing the weights for basis polynomials
	// necessary for evaluation of the polynomials at arbitary points
	for i := 0; i < n; i++ {
		weights[i] = 1.0
		for j := 0; j < n; j++ {
			if i != j {
				if math.Abs(xPoints[i]-xPoints[j]) < 1e-9 {
					return nil, fmt.Errorf("duplicate xPoint detected : x[%d] = x[%d] = %f", i, j, xPoints[i])
				}
				weights[i] *= 1.0 / (xPoints[i] - xPoints[j])
			}
		}
	}

	return &LagrangePolynomial{xPoints, yPoints, weights}, nil
}

// interpolate computes the interpolated value at a single x
func (lp *LagrangePolynomial) Interpolate(x float64) float64 {
	n := len(lp.xPoints)
	result := 0.0

	for i := 0; i < n; i++ {
		term := lp.yPoints[i] * lp.weights[i]
		for j := 0; j < n; j++ {
			if i != j {
				term *= (x - lp.xPoints[j])
			}
		}
		result += term
	}

	return result
}

// calculates the interpolated values for multiple x values
func (lp *LagrangePolynomial) InterpolateMany(xValues []float64) []float64 {
	results := make([]float64, len(xValues))
	for i, x := range xValues {
		results[i] = lp.Interpolate(x)
	}
	return results
}

func main() {
	xPoints := []float64{1, 2, 3}
	yPoints := []float64{1, 4, 9}

	lagrange, err := NewLagrangePolynomial(xPoints, yPoints)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}

	x := 2.5
	y := lagrange.Interpolate(x)
	fmt.Printf("The interpolated value at x = %.2f is y = %.2f\n", x, y)

	xValues := []float64{1.5, 2.5, 3.5}
	yValues := lagrange.InterpolateMany(xValues)
	fmt.Println("Interpolated values:")
	for i, xv := range xValues {
		fmt.Printf("x = %.2f, y = %.2f\n", xv, yValues[i])
	}
}
