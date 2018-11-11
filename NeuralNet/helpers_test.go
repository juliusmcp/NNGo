package NeuralNet

import (
	"gonum.org/v1/gonum/mat"
	"testing"
	
)

func TestToFloatSlice(t *testing.T) {

	weights := []float64{
		0.1, 0.12, 0.8,  0.17, 0.2, 0.25,
	}
	matrix:= mat.NewDense(6,1, weights)
	backtoslice:=ToFloatSlice(matrix)
	slicelength:=len(backtoslice)
	if slicelength!=6 {
		t.Error("TestToFloatSlice: Expected 6, got ", slicelength)	
	}

}
func TestToFloatSliceDiffShape(t *testing.T) {

	weights := []float64{
		0.1, 0.12, 0.8,  0.17, 0.2, 0.25,
	}
	matrix:= mat.NewDense(3,2, weights)
	backtoslice:=ToFloatSlice(matrix)
	slicelength:=len(backtoslice)
	if slicelength!=6 {
		t.Error("TestToFloatSliceDiffShape Expected 6, got ", slicelength)	
	}

}

func TestToMultiFloatSlice(t *testing.T) {

	weights := []float64{
		0.1, 0.12, 0.8,  0.17, 0.2, 0.25,
	}
	matrix:= mat.NewDense(3,2, weights)
	backtoslice:=ToFloatMultiSlice(matrix)
	slicelength:=len(backtoslice)
	if slicelength!=3 {
		t.Error("TestToMultiFloatSlice Expected 3, got ", slicelength)	
	}

	for i := 0; i < 3; i++ {
		for j := 0; j < 2; j++ {
			if backtoslice[i][j]!= matrix.At(i,j){
				t.Error("TestToMultiFloatSlice jumbled results")	
			}
		}
	}
		
}


func TestNewMatrix(t *testing.T) {

	data := [][]float64{
		{1, 2},
		{3, 4},
		{5, 6},
	
	}
	nm:=NewMatrixM(data,false)
	or,oc:=nm.Dims()
	if or!=3 {
		t.Error("TestNewMatrix Expected 3 rows, got ", or)	
	}
	if oc!=2 {
		t.Error("TestNewMatrix Expected 2 columns, got ", oc)	
	}
	r:=nm.At(1,0)
	if r!=3{
		t.Error("TestNewMatrix Expected 3 at row 2 col 2, got ", r)	
	}
	
		
}
func TestNewMatrixReverse(t *testing.T) {

	data := [][]float64{
		{1, 2},
		{3, 4},
		{5, 6},
	
	}
	nm:=NewMatrixM(data,true)
	or,oc:=nm.Dims()
	if or!=2 {
		t.Error("TestNewMatrixReverse Expected 2 rows, got ", or)	
	}
	if oc!=3 {
		t.Error("TestNewMatrixReverse Expected 3 columns, got ", oc)	
	}
	r:=nm.At(1,0)
	if r!=2{
		t.Error("TestNewMatrix Expected 2 at row 2 col 2, got ", r)	
	}
	
		
}