package main

import (
	"math"

	"gonum.org/v1/gonum/mat"
	// "google.golang.org/genproto/googleapis/cloud/aiplatform/v1/schema/predict/prediction"
	// "gonum.org/v1/stat"
)

type LogisticRegression struct {
	weights *mat.Dense
}

func NewLogisticRegression() *LogisticRegression {
	return &LogisticRegression{
		weights: mat.NewDense(1, 1, nil),
	}
}

func sigmoid(z float64) float64 {
	return 1.0 / (1.0 + math.Exp(-z))
}

func (lr *LogisticRegression) Train(X, y *mat.Dense, learningRate float64, iterations int) {
	_, numFeatures := X.Dims()
	lr.weights = mat.NewDense(1, numFeatures, nil)

	for i := 0; i < iterations; i++ {
		var predictions mat.Dense
		predictions.Mul(X, lr.weights.T())
		predictions.Apply(func(_, _ int, v float64) float64 { return sigmoid(v) }, &predictions)

		var errors mat.Dense
		errors.Sub(&predictions, y)

		var gradient mat.Dense
		gradient.Mul(errors.T(), X)
		gradient.Scale(learningRate/float64(len(y.RawMatrix().Data)), &gradient)
	}
}

func (lr *LogisticRegression) Predict(X *mat.Dense) *mat.Dense {
	var predictions mat.Dense
	predictions.Mul(X, lr.weights.T())
	predictions.Apply(func(_, _ int, v float64) float64 { return sigmoid(v) }, &predictions)

	return &predictions
}
