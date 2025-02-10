package main

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

func main() {

}

func generateData(n int) (*mat.Dense, *mat.Dense) {
	X := mat.NewDense(n, 2, nil)
	y := mat.NewDense(n, 1, nil)

	for i := 0; i < n; i++ {
		X.Set(i, 0, rand.NormFloat64())
		X.Set(i, 1, rand.NormFloat64())
	}
}
