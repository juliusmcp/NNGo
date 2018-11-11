package NeuralNet

import (
	"bytes"
	"fmt"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
	"io"
	"math"
	
)


type Transformation struct {
	Name					string
	Inputs					*mat.Dense
	Outputs					*mat.Dense
	Weights  				*mat.Dense
	DeltaWeights			*mat.Dense
	dTotErrWRTotNetIn	    *mat.Dense
	InputCount	 			int
	OutputCount  			int
	Bias        			bool
	ActivationFunc			func(r, c int, z float64) float64
	prevLayer, nextLayer   *Transformation
	index       			int
	
}

func NewTransformation(inputs int, bias bool, outputs int) *Transformation{

	return &Transformation{
		InputCount: 	inputs,
		Bias: 			bias,
		OutputCount: 	outputs,
	
	}
}
func NewNamedTransformation(inputs int, bias bool, outputs int, name string) *Transformation{

	return &Transformation{
		InputCount: 	inputs,
		Bias: 			bias,
		OutputCount: 	outputs,
		Name: 			name,
	}
}
//Initialise - Initialise with Random Weights
func (transformation *Transformation) Initialise() {
	icount:= transformation.InputCount
	ocount:= transformation.OutputCount
	if transformation.Bias {
		icount++
	}
	data:= random((icount)*ocount, float64(icount))
	transformation.Weights = NewMatrix(ocount, icount, data)

}
//InitialiseWithWeights - Initialise with starting weights
func (transformation *Transformation) InitialiseWithWeights(weights []float64) {
	icount:= transformation.InputCount
	ocount:= transformation.OutputCount
	if transformation.Bias {
		icount++
	}
	if weights==nil {
		transformation.Weights =nil
	}
	transformation.Weights = NewMatrix(ocount, icount, weights)
}

func (transformation *Transformation) UpdateWeights(deltas [][]float64, learningRate float64) string {
	//transformation.Weights=
	var buffer bytes.Buffer
	outputstr:="Weights Adjusted \n %v\n\n"
	d:=NewMatrixM(deltas,false)
	d.Scale(learningRate,d)
	
	outputs:= fmt.Sprintf(outputstr, mat.Formatted(transformation.Weights, mat.Prefix("Outputs"), mat.Excerpt(0)))
	fmt.Fprintf(&buffer, outputs)
	transformation.Weights.Sub(transformation.Weights,d)

	outputs= fmt.Sprintf(outputstr, mat.Formatted(transformation.Weights, mat.Prefix("Outputs"), mat.Excerpt(0)))
	fmt.Fprintf(&buffer, outputs)
	return buffer.String()

}
//GetWeights - Return Transformation Weights as float64 slice
func (transformation *Transformation) GetWeights() []float64 {
	return ToFloatSlice(transformation.Weights)
}
//GetTransformationInfo
func (transformation *Transformation) GetTransformationInfo(trace io.Writer) string {
	outputstr:="Inputs %d\nBias %t\nOutputs %d"
	output:= fmt.Sprintf(outputstr,transformation.InputCount,transformation.Bias,transformation.OutputCount)

	if (trace!=nil) {
	   fmt.Fprintf(trace, output)
	   
	}
	return output
}
//VisualiseWeights
func (transformation *Transformation) VisualiseWeights(trace io.Writer) string {

	 outputstr:="Weights :\n\n         Inputs\nOutputs%v\n\n"
	 output:= fmt.Sprintf(outputstr, mat.Formatted(transformation.Weights, mat.Prefix("Outputs"), mat.Excerpt(0)))
	 if (trace!=nil) {
		fmt.Fprintf(trace, output)
	 }
	 return output
}
func (transformation *Transformation) VisualiseTransformation(trace io.Writer) string {

	outputstr:="Inputs: \n%v\nWeights : \n%v\nOutputs: \n%v\nDeltaWeights : \n%v\n"
	output:= fmt.Sprintf(outputstr,
				 mat.Formatted(transformation.Inputs, mat.Prefix(""), mat.Excerpt(0)),
				 mat.Formatted(transformation.Weights, mat.Prefix(""), mat.Excerpt(0)),
				 mat.Formatted(transformation.Outputs, mat.Prefix(""), mat.Excerpt(0)),
				 mat.Formatted(transformation.DeltaWeights, mat.Prefix(""), mat.Excerpt(0)))
	
	if (trace!=nil) {
	   fmt.Fprintf(trace, output)
	}
	return output
}
//SetActivateSigmoid Set activation to Sigmoid
func (transformation *Transformation) SetActivateSigmoid() {

	 transformation.SetActivationFunc(Sigmoid)
}

//SetActivationFunc Set activation func
func (transformation *Transformation) SetActivationFunc(activationFunc func(r, c int, z float64) float64) {
	transformation.ActivationFunc=activationFunc 
}

//Activate Process forward pass of Transformation with Supplied Activation Func
func (transformation *Transformation) Activate(inputs []float64, trace io.Writer) []float64 {
	if (transformation.Bias && len(inputs)!= transformation.InputCount+1) {
		if trace !=nil {
			fmt.Fprintf(trace, "Incorrect Number of inputs. Attempt to Add Bias")
		}
		inputsPlusBias:=AddBiasToInputs(inputs)
		transformation.Inputs = NewMatrix(len(inputsPlusBias), 1, inputsPlusBias)
	} else {
		transformation.Inputs = NewMatrix(len(inputs), 1, inputs)
	}
	
	if trace !=nil {
		fmt.Fprintf(trace, "Transformation Inputs: \n%v\n\n", mat.Formatted(transformation.Inputs, mat.Prefix(""), mat.Excerpt(0)))
		fmt.Fprintf(trace, "Transformation Weights: \n%v\n\n", mat.Formatted(transformation.Weights, mat.Prefix(""), mat.Excerpt(0)))
	}
	r, _ := transformation.Weights.Dims()
	_, c := transformation.Inputs.Dims()
	inputsAfterWeights := NewMatrix(r, c, nil)
	inputsAfterWeights.Product(transformation.Weights, transformation.Inputs)
	if trace !=nil {
		fmt.Fprintf(trace, "Weighted :\n%v\n\n", mat.Formatted(inputsAfterWeights, mat.Prefix(""), mat.Excerpt(0)))
	}
	if transformation.ActivationFunc!=nil{
		transformation.Outputs = ActivationFunc(inputsAfterWeights, transformation.ActivationFunc).(*mat.Dense)
	} else {
		transformation.Outputs = inputsAfterWeights
	}
	if trace !=nil {
		fmt.Fprintf(trace , "Transformation Outputs :\n%v\n\n", mat.Formatted(transformation.Outputs, mat.Prefix(""), mat.Excerpt(0)))
	}
	return ToFloatSlice(transformation.Outputs)
}


func random(size int, v float64) (data []float64) {
	dist := distuv.Uniform{
		Min: -1 / math.Sqrt(v),
		Max: 1 / math.Sqrt(v),
	}

	data = make([]float64, size)
	for i := 0; i < size; i++ {
		data[i] = dist.Rand()
	}
	return
	//poss rand.NormFloat64()
}
