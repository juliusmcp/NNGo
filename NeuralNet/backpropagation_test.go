package NeuralNet

import (
	//"fmt"
	//"gonum.org/v1/gonum/mat"
	"testing"
	
)

func TestBackPropExampleOne (t *testing.T) {
	
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

	hiddenToOutputtrainer:= NewBackPropagation(SigmoidDerivative)
	inputToHiddentrainer:= NewBackPropagation(SigmoidDerivative)

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
	wi, _:=hiddenToOutputtrainer.CalculateGradientsOutputLayer(hiddenToOutput,target,nil)
	wirow:=len(wi)
	if wirow!=1{
		t.Error("TestBackPropExampleOne Back Prop Output layer Expected 1 delta weights, got ", wirow)	
	}
	iwdelta:=[][]float64{
		{-0.08864282287706811,-0.08007772468295891,-0.10646805643473987},
	}
	for i := 0; i < wirow; i++ {
		  wicol:=wi[i]
		  iwdelcol:=iwdelta[i]
		  for j := 0; j < len(wi[i]); j++ {

			if wicol[j]!=iwdelcol[j] {
				t.Error("TestBackPropExampleOne outer weight deltas don't match")
			}
		  }
		  
		  
    }
	
	wo, _:=inputToHiddentrainer.CalculateGradientsHiddenLayer(inputToHiddenLayer,hiddenToOutput,nil)
	worow:=len(wo)
	if 	worow!=2{
		t.Error("TestBackPropExampleOne Back Prop Hidden layer Expected 6 delta weights, got ", worow)	
	}
	owdelta:=[][]float64{
		{0  ,-0.000964656816362879, -0.000964656816362879},
		{0  ,-0.0030964341362857482, -0.0030964341362857482},
	
	}
	for i := 0; i < worow;i++ {
		  wocol:=wo[i]
		  owdelcol:=owdelta[i]
		  for j := 0; j < len(wo[i]); j++ {

			if wocol[j]!=owdelcol[j] {
				t.Error("TestBackPropExampleOne outer weight deltas don't match")
			}
		  }
		  
		  
    }
	

}