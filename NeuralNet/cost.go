package NeuralNet

import (

	"gonum.org/v1/gonum/mat"
	//"math"
)



func MeanError(errorData []float64) float64 {
	noOfTrainingExamples:=len(errorData)
	errors := mat.NewDense(noOfTrainingExamples, 1, errorData)
	sum:= mat.Sum(errors)
	a:= float64(1)/float64(noOfTrainingExamples)
	return a * sum
}

/*
func SquaredErrorMatrix(outputData mat.Matrix, targetData []float64) float64 {
	targets := mat.NewDense(len(targetData), 1, targetData)
	r, c := outputData.Dims()
	o := mat.NewDense(r, c, nil)
	o.Sub(targets, outputData)
	o.MulElem(o,o)
	return mat.Sum(o)
}

func SquaredError(outputData []float64, targetData []float64) float64 {
	output:= mat.NewDense(len(outputData), 1, outputData)
	return SquaredErrorMatrix(output,targetData)
}

func SquaredErrorDerivative(outputs *mat.Dense, targets []float64) *mat.Dense{
	o := mat.NewDense(len(targets), 1, targets)
	o.Sub(outputs, o)
	di := mat.NewDense(1, 1, []float64{2.0})
	o.MulElem(o,di) 
	return o
}*/

func SquaredError(output []float64, target []float64) []float64 {

	t :=NewMatrix(len(target), 1, target)
	input := make([]float64, len(output))
	copy(input, output)
	o := mat.NewDense(len(output), 1, input)
	o.Sub(t, o)
	o.MulElem(o,o)
    return ToFloatSlice(o)
}

func SquaredErrorSum(output []float64, target []float64) float64 {
	t := NewMatrix(len(target), 1, target)
	o := NewMatrix(len(output), 1, output)
	o.Sub(t, o)
	o.MulElem(o,o)
	return mat.Sum(o)
}



