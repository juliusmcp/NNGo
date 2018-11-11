package NeuralNet

import (
	
	"errors"
	"fmt"
	"gonum.org/v1/gonum/mat"
	"io"
	/*"fmt"
	"gonum.org/v1/gonum/stat/distuv"
	"io"
	"math"*/
	
)


type BackPropagation struct {
		deriativeActivation func(r, c int, z float64) float64
		//deriativeCost       func(r, c int, z float64) float64
}


func NewBackPropagation(deriativeActivation func(r, c int, z float64) float64) BackPropagation{
	return BackPropagation{
		deriativeActivation: deriativeActivation,
		//deriativeCost: deriativeCost,

	}
}

func (backprop BackPropagation) CalculateGradientsOutputLayer(transformation *Transformation, targets []float64, trace io.Writer) ([][]float64, error) {
	
	if transformation.Weights==nil {
		return nil,  errors.New("no weight matrix")
	}
	if transformation.Inputs==nil {
		return nil, errors.New("no input matrix")
	}
	if transformation.Outputs==nil {
		return nil, errors.New("no output matrix")
	}
	//1)
	//---
	//- - target sub output
	//---
	//- - target sub output
	//---

	//par d cost output
	e := NewMatrix(len(targets), 1, targets)
	e.Sub(transformation.Outputs,e)
	if trace !=nil {
		fmt.Fprintf(trace, "##1 dOutput wr dTotalInput :\n%v\n\n",  mat.Formatted(e, mat.Prefix(""), mat.Excerpt(0)))
	}
	//2)
	//---
	//- - output sub der activation
	//---
	//- - output sub der activation
	//---
	//par d output activation
	da:= mat.DenseCopyOf(transformation.Outputs)
	da.Apply(backprop.deriativeActivation,da)
	if trace !=nil {
		fmt.Fprintf(trace, "##2 delta Total error wr delta Ouput :\n%v\n\n",  mat.Formatted(da, mat.Prefix(""), mat.Excerpt(0)))
	}
	//---
	//- - 1 mul 2
	//---
	//- - 1 mul 2
	//---
	da.MulElem(e, da)
	if trace !=nil {
		fmt.Fprintf(trace, "##3 delta TotalError wr delta TotalNetInput :\n%v\n\n",  mat.Formatted(da, mat.Prefix(""), mat.Excerpt(0)))
	}


	transformation.dTotErrWRTotNetIn=da
	//------
	//- -- -
	//------
	transformation.DeltaWeights= mat.DenseCopyOf(transformation.Weights)
	//------		 ---
	//- -- -    *    - - inputs
	//------		 ---
	//				 ---
	//				 - -
	//	 			 ---
	//				 ---
	//				 - -
	//	 			 ---
	transformation.DeltaWeights.Mul(da,Reshape(transformation.Inputs))

	if trace !=nil {
		fmt.Fprintf(trace, "##4 Weights * output :\n%v\n\n",  mat.Formatted(transformation.DeltaWeights, mat.Prefix(""), mat.Excerpt(0)))
	}

	return ToFloatMultiSlice(transformation.DeltaWeights), nil


	
}
func (backprop BackPropagation) CalculateGradientsHiddenLayer(transformation *Transformation, previous *Transformation, trace io.Writer) ([][]float64, error){
	//par d cost output

	if transformation.Weights==nil {
		return nil,  errors.New("no weight matrix")
	}
	if transformation.Inputs==nil {
		return nil, errors.New("no input matrix")
	}
	if transformation.Outputs==nil {
		return nil, errors.New("no putput matrix")
	}
	if previous.dTotErrWRTotNetIn==nil {
		return nil, errors.New("no delta totalerror with regard delta totalnetinput set previous transformation")
	}


	pw:=getPreviousWeightsForBackProp(previous)
	e := NewMatrix(transformation.OutputCount, 1, nil)
	e.Mul(Reshape(pw),previous.dTotErrWRTotNetIn)
	
	


	if trace !=nil {
		fmt.Fprintf(trace, "##I1 delta Total error wr Hidden Output :\n%v\n\n",  mat.Formatted(e, mat.Prefix(""), mat.Excerpt(0)))
	}
	
	da:= mat.DenseCopyOf(transformation.Outputs)

	da.Apply(backprop.deriativeActivation,da)
	if trace !=nil {
		fmt.Fprintf(trace, "##I2 Hidden Output wr total net input :\n%v\n\n",  mat.Formatted(da, mat.Prefix(""), mat.Excerpt(0)))
	}
	

	da.MulElem(e, da)
	transformation.dTotErrWRTotNetIn=da
	if trace !=nil {
		fmt.Fprintf(trace, "##I3 delta TotalError wr delta TotalNetInput :\n%v\n\n",  mat.Formatted(da, mat.Prefix(""), mat.Excerpt(0)))
	}


	wr, wc:=transformation.Weights.Dims()
	dw:= NewMatrix(wr, wc, nil)
	di:=transformation.Inputs.T()
	dw.Mul(da,di)
	transformation.DeltaWeights=dw
	return ToFloatMultiSlice(dw), nil
	
}



func getPreviousWeightsForBackProp(previous *Transformation) *mat.Dense{
	if (!previous.Bias) {
		return mat.DenseCopyOf(previous.Weights)
	}
	pwr, pwc:=previous.Weights.Dims()
	if (previous.Bias) {
		pwc--
	}
	data:= ToResizedFloatSlice(pwr, pwc, previous.Weights)
	return NewMatrix(pwr, pwc,data)

}