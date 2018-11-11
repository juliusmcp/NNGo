package NeuralNet


import (

	"math"
	"gonum.org/v1/gonum/mat"
	//"gonum.org/v1/gonum/stat/distuv"
	//"strconv"
	//"time"
)

//Apply activation function
func ActivationFunc(m mat.Matrix, activation func(r, c int, z float64) float64) mat.Matrix {
	if (activation==nil) {
		return m
	}
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Apply(activation,m)
	return o
}
func DerivativeActivationFunc(m mat.Matrix, derActivation func(r, c int, z float64) float64) mat.Matrix {
	if (derActivation==nil) {
		return m
	}
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Apply(derActivation,m)
	return o
}
func Sigmoid(r, c int, z float64) float64 {
	return 1.0 / (1 + math.Exp(-1*z))
}

func SigmoidDerivative(r, c int, z float64) float64 {
	return z * (1 - z)
}
