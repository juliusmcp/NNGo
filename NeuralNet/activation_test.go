package NeuralNet

import (
	//"fmt"
	"gonum.org/v1/gonum/mat"
	"testing"
	
)

func TestSigmoidOutput(t *testing.T) {
	
	inputs:=[]float64{0.5}
	expected:=[]float64{0.6224593312018546}
	matrix:= mat.NewDense(1, 1, inputs)
	result:= ActivationFunc(matrix,Sigmoid)
	expectedmatrix:= mat.NewDense(1, 1, expected)
	if !mat.Equal(result,expectedmatrix){
		t.Error("TestSigmoidOutput: Expected 0.622459331201, got ", result)	
	}

}
func TestSigmoidTwoOutput(t *testing.T) {
	
	inputs:=[]float64{3,0.1}
	expected:=[]float64{0.9525741268224334,0.52497918747894}
	matrix:= mat.NewDense(2, 1, inputs)
	result:= ActivationFunc(matrix,Sigmoid)
	expectedmatrix:= mat.NewDense(2, 1, expected)
	if !mat.Equal(result,expectedmatrix){
		t.Error("TestSigmoidTwoOutput(: Expected 0.9525741268224334,0.52497918747894, got ", result)	
	}
	
}
