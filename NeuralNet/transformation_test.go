package NeuralNet

import (
	"fmt"
	//"gonum.org/v1/gonum/mat"

	"testing"

	
)


func TestInitialiseWithOutBias(t *testing.T) {
	transform:=NewTransformation(2, false, 2)
	transform.Initialise()
	r, c:=transform.Weights.Dims()
	if r!=2 {
		t.Error("TestInitialiseWithOutBias Expected row count of 2, got ", r)	
	}
	if c!=2 {
		t.Error("TestInitialiseWithOutBias Expected column count of 2, got ", c)	
	}
	selected:= transform.Weights.At(0,0)
	if selected==0.0 {
		transform.Initialise()
		if transform.Weights.At(0,0)==0.0 {
			t.Error("TestInitialiseWithOutBias data not randomised.")	
		}
	}


}
func TestInitialiseWithBias(t *testing.T) {
	transform:=NewTransformation(2, true, 2)
	transform.Initialise()
	r, c:=transform.Weights.Dims()
	if r!=2 {
		t.Error("TestInitialiseWithOutBias Expected row count of 2, got ", r)	
	}
	if c!=3 {
		t.Error("TestInitialiseWithOutBias Expected column count of 3, got ", c)	
	}
	if transform.Weights.At(0,0)==0.0 {
		transform.Initialise()
		if transform.Weights.At(0,0)==0.0 {
			t.Error("TestInitialiseWithOutBias data not randomised.")	
		}
	}
}

func TestInitialiseWithWeight(t *testing.T) {
	transform:=NewTransformation(2, true, 2)
	weights:= []float64{
		0.0, 0.0, 1.0, 0.0, 1.0, 1.0,
	}
	transform.InitialiseWithWeights(weights)
	r, c:=transform.Weights.Dims()
	if r!=2 {
		t.Error("TestInitialiseWithOutBias Expected row count of 2, got ", r)	
	}
	if c!=3 {
		t.Error("TestInitialiseWithOutBias Expected column count of 3, got ", c)	
	}
	selected:= transform.Weights.At(1,1)
	selectedbias:= transform.Weights.At(1,2)
	if selected!=1.0 {
		t.Error("TestInitialiseWithWeight : expect 1, got ",selected)	
	}
	if selectedbias!=1.0 {
		t.Error("TestInitialiseWithWeight : expect 1, got ",selectedbias)	
	}
}

func TestForwardPass(t *testing.T) {

	data := []float64{
		 0.15,0.35,
	}
	expected:= []float64{
		0.7020334875314546,0.5855259169748928,
    }
	transform:=NewTransformation(2, true, 2)
	weights:= []float64{
		0.1, 0.12, 0.8,  0.17, 0.2, 0.25,
	}
	transform.InitialiseWithWeights(weights)
	fmt.Print(transform.VisualiseWeights(nil))
	transform.SetActivateSigmoid()
	outputs:=transform.Activate(data, nil)

	for i := range outputs {
        if expected[i] != outputs[i] {
			t.Error("TestForwardPass")	
        }
    }

}


func Test3LayersForward(t *testing.T) {

	data := []float64{
		 0.15,0.35,
	}
	expected:= []float64{
		0.7191938702879944,0.6192699938189198,
    }
	transformToHidden:=NewTransformation(2, true, 2)
	hweights:= []float64{
		0.1, 0.12, 0.8,  0.17, 0.2, 0.25,
	}
	transformToHidden.InitialiseWithWeights(hweights)
	transformToHidden.SetActivateSigmoid()
	fmt.Print(transformToHidden.VisualiseWeights(nil))
	transformToOutput:=NewTransformation(2, true, 2)
	oweights:= []float64{
		0.05, 0.33, 0.15,  0.4, 0.07, 0.7,
	}
	transformToOutput.InitialiseWithWeights(oweights)
	transformToOutput.SetActivateSigmoid()
	fmt.Print(transformToOutput.VisualiseWeights(nil))
	firstoutputs:=transformToHidden.Activate(data, nil)
	secondoutputs:=transformToOutput.Activate(firstoutputs, nil)
	for i := range secondoutputs {
        if expected[i] != secondoutputs[i] {
			t.Error("Test3LayersForward")	
        }
    }

}


func TestForwardExampleOne(t *testing.T){
	data:=[]float64{0.0,1.0}
	target:=[]float64{1.0}
	inputToHiddenLayer:=NewNamedTransformation(2, true, 2,"Input->Hidden")
	testweightsone:= []float64{0.604,0.940,0.664,0.437,0.424,0.686}
	inputToHiddenLayer.InitialiseWithWeights(testweightsone)
	inputToHiddenLayer.SetActivateSigmoid()
	hiddenToOutput:=NewNamedTransformation(2, true, 1,"Hidden->Output")
	testweightstwo:= []float64{0.065,0.156,0.096}
	hiddenToOutput.InitialiseWithWeights(testweightstwo)
	hiddenToOutput.SetActivateSigmoid()

		p:= AddBiasToInputs(data)
	hiddenoutputs:=inputToHiddenLayer.Activate(p, nil)
	if len(hiddenoutputs)!=2 {
		t.Error("TestBackPropExampleOne Hidden Outputs: Expect 2 outputs got.", len(hiddenoutputs))
	}
	if hiddenoutputs[0]!=0.8325766980765932 {
		t.Error("TestBackPropExampleOne Hidden Outputs: Expect result 1=0.8325766980765932, got ", hiddenoutputs[0])	
	}
	if hiddenoutputs[1]!=0.7521291114395702 {
		t.Error("TestBackPropExampleOne Hidden Outputs: Expect result 2= 0.7521291114395702, got ", hiddenoutputs[0])	
	}
    hiddenoutputs=AddBiasToInputs(hiddenoutputs)
	outputs:= hiddenToOutput.Activate(hiddenoutputs, nil)
	if len(outputs)!=1 {
		t.Error("TestBackPropExampleOne Final Output: Expect 1 output got.", len(outputs))
	}
	if outputs[0]!=0.5664666852388589{
		t.Error("TestBackPropExampleOne Final Output: Expect result 0.5664666852388589, got ", outputs[0])	
	}
	localError:= SquaredErrorSum(outputs,target)
	if localError!=0.18795113500778265{
		t.Error("TestBackPropExampleOne localError:: Expect result 0.187951, got ", localError)	
	}
}
func TestWeightUpdate(t *testing.T) {
	odelta:=[][]float64{
		{-0.08864282287706811,-0.08007772468295891,-0.10646805643473987},
	}
	delta:=[][]float64{
		{0  ,-0.000964656816362879, -0.000964656816362879},
		{0  ,-0.0030964341362857482, -0.0030964341362857482},
	
	}
	afterodelta:=[][]float64{
		{0.1359142583016545,0.22006217974636713,0.1811744451477919},
	}
	afterdelta:=[][]float64{
		{0.604  ,0.9407717254530903, 0.6647717254530904},
		{0.437  ,0.4264771473090286, 0.6884771473090286},
	
	}
	inputToHiddenLayer:=NewNamedTransformation(2, true, 2,"Input->Hidden")
	testweightsone:= []float64{0.604,0.940,0.664,0.437,0.424,0.686}
	inputToHiddenLayer.InitialiseWithWeights(testweightsone)
	hiddenToOutput:=NewNamedTransformation(2, true, 1,"Hidden->Output")
	testweightstwo:= []float64{0.065,0.156,0.096}
	hiddenToOutput.InitialiseWithWeights(testweightstwo)
	
	inputToHiddenLayer.UpdateWeights(delta,0.8)
	newweightsih:=inputToHiddenLayer.Weights
	for i := 0; i < len(afterdelta); i++ {
		col:=afterdelta[i]
		for j := 0; j < len(col); j++ {

		  if col[j]!=newweightsih.At(i,j) {
			  t.Errorf("TestWeightUpdate first transformation weight after change, expect %f, got %f",col[j],newweightsih.At(i,j))
		  }
		}
		
		
   }
	hiddenToOutput.UpdateWeights(odelta,0.8)
	newweightsio:=hiddenToOutput.Weights
	for i := 0; i < len(afterodelta); i++ {
		col:=afterodelta[i]
		for j := 0; j < len(col); j++ {

		  if col[j]!=newweightsio.At(i,j) {
			  t.Errorf("TestWeightUpdate second transformation weight after change, expect %f, got %f",col[j],newweightsio.At(i,j))
		  }
		}
		
		
   }
	
}