package NeuralNet

import (
	//"fmt"
	//"gonum.org/v1/gonum/mat"
	"testing"
	
)

func TestLocalErrorOneOutput(t *testing.T) {
	
	target:=[]float64{5}
	output:=[]float64{2}
	expected:=[]float64{9}
	

	result:=SquaredError(output,target)
	if result[0]!=expected[0]{
		t.Error("TestLocalErrorOneOutput: Expected 9, got ", result)	
	}
}
func TestLocalErrorTwoOutput(t *testing.T) {
	
	target:=[]float64{1,0.5}
	output:=[]float64{0.1,0.05}
	expected:=[]float64{1.0125000000000002}

	result:=SquaredErrorSum(output,target)
	if result!=expected[0]{
		t.Error("TestLocalErrorTwoOutput: Expected 0.81,0.2025, got ", result)	
	}
	
}

