package NeuralNet

import (
	
	"gonum.org/v1/gonum/mat"

)

func Reshape(input *mat.Dense) *mat.Dense{
	r, c := input.Dims()
	
	res:=make([]float64,0)
		for j := 0; j < c; j++ {
			for i := 0; i < r; i++ {
				res=append(res,input.At(i,j))
			}
			
	}
	da:= mat.NewDense(c,r,res)
	return da
}

func ToFloatSlice(input *mat.Dense) []float64{
	r, c := input.Dims()
	res:=make([]float64,0)
	for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				res=append(res,input.At(i,j))
			}
			
	}
	return res
}

func ToResizedFloatSlice(r, c int, input *mat.Dense) []float64{
	res:=make([]float64,0)
	for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				res=append(res,input.At(i,j))
			}
			
	}
	return res
}
func ToFloatMultiSlice(input *mat.Dense) [][]float64{
	r, c := input.Dims()
	res:=make([][]float64,r)
	for i := 0; i < r; i++ {
		  inner:=make([]float64,c)  
			for j := 0; j < c; j++ {
				inner[j] =input.At(i,j)
			}
			res[i]=inner
			
	}
	return res
}

func ToResizedFloatMultiSlice(r, c int, input *mat.Dense) [][]float64{
	res:=make([][]float64,r)
	for i := 0; i < r; i++ {
		  inner:=make([]float64,c)  
			for j := 0; j < c; j++ {
				inner[j] =input.At(i,j)
			}
			res[i]=inner
			
	}
	return res
}
//AddBiasToInputs Bias to Input slice
func AddBiasToInputs(Inputs []float64) []float64 {
	return append(Inputs,1)
}

func NewMatrix(r, c int, data []float64) *mat.Dense {
	if data==nil {
		return mat.NewDense(r,c, nil)
	}
	input := make([]float64, len(data))
	copy(input, data)
	return mat.NewDense(r,c, input)
}
//NewMatrixM converts multi dimension slice to matrix, rows/columns if revers columns/rows
func NewMatrixM(input [][]float64, reverse bool) *mat.Dense {

	r:= len(input)
	c:=0
	if (r>0){
		c=len(input[0])
	}
	
	if reverse {
		nm:=mat.NewDense(c,r,nil)
		for i,sl:= range input{
			data := make([]float64, len(sl))
			copy(data ,sl)
			nm.SetCol(i,data)
		}
		return nm
	} else {
		nm:=mat.NewDense(r,c,nil)
		for i,sl:= range input{
			data := make([]float64, len(sl))
			copy(data ,sl)
			nm.SetRow(i,data)
		}
		return nm
	}
	
}

func NewMatrixFill(r,c int, fill float64) *mat.Dense {
	cells:= r*c
	o := make([]float64, cells)
	for i := range o {
		o[i] = 1
	}
	return mat.NewDense(r, c, o)
}
