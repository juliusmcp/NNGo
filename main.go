package main

import (
	"fmt"
	"NNGo/NeuralNet"
	"os"
)

func main() {
	TrainingThree()
}


func Training() {
	data := [][]float64{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}
	target := [][]float64{
		{0}, {1}, {1}, {0},
	}

	testweightsone:= []float64{0.604,0.940,0.664,0.437,0.424,0.686}
	testweightstwo:= []float64{0.065,0.156,0.096}
	inputToHiddenLayer:=NeuralNet.NewTransformation(2, true, 2)
	inputToHiddenLayer.InitialiseWithWeights(testweightsone)
	//inputToHiddenLayer.Initialise()
	inputToHiddenLayer.SetActivateSigmoid()

	hiddenToOutput:=NeuralNet.NewTransformation(2, true, 1)
	//hiddenToOutput.Initialise()
	hiddenToOutput.InitialiseWithWeights(testweightstwo)
	hiddenToOutput.SetActivateSigmoid()

	hiddenToOutputtrainer:= NeuralNet.NewBackPropagation(NeuralNet.SigmoidDerivative)
	inputToHiddentrainer:= NeuralNet.NewBackPropagation(NeuralNet.SigmoidDerivative)

	
    lasttserror:=make([]float64,4)
    for epoch:=0;epoch<1000;epoch++ {
		errorAccm:=[]float64{}
		
		for i, p := range data {
			
			in:= NeuralNet.AddBiasToInputs(p)
			hiddenoutputs:=inputToHiddenLayer.Activate(in, nil)
			hiddenoutputs=NeuralNet.AddBiasToInputs(hiddenoutputs)
			outputs:= hiddenToOutput.Activate(hiddenoutputs, nil)
			localError:= NeuralNet.SquaredErrorSum(outputs,target[i])
			var errchange float64
			if (epoch>0) {
				errchange=localError-float64(lasttserror[i])
			}
			lasttserror[i]=localError
			fmt.Printf("Actual %f Target %f Error %f (Change %f)\n" ,outputs[0],target[i],localError,errchange)
			errorAccm=append(errorAccm,localError)

			wi, _:=hiddenToOutputtrainer.CalculateGradientsOutputLayer(hiddenToOutput,target[i],nil)
			wo, _:=inputToHiddentrainer.CalculateGradientsHiddenLayer(inputToHiddenLayer,hiddenToOutput,nil)
			//hiddenToOutput.VisualiseWeights(os.Stdout)
			//inputToHiddenLayer.VisualiseWeights(os.Stdout)
			hiddenToOutput.UpdateWeights(wi,0.8)
			inputToHiddenLayer.UpdateWeights(wo,0.8)
			
		}
			
		epochError:=NeuralNet.MeanError(errorAccm)
		fmt.Printf("EpochError %d: %f\n" ,epoch,epochError)
	}
	
}

func TrainingTest() {

	data:=[]float64{0.0,1.0}
	target:=[]float64{1.0}
	inputToHiddenLayer:=NeuralNet.NewNamedTransformation(4, true, 3,"Input->Hidden")
	testweightsone:= []float64{0.604,0.940,0.664,0.437,0.424,0.686}

	inputToHiddenLayer.InitialiseWithWeights(testweightsone)
	inputToHiddenLayer.SetActivateSigmoid()
	inputToHiddenLayer.VisualiseWeights(os.Stdout)

	hiddenToOutput:=NeuralNet.NewNamedTransformation(2, true, 1,"Hidden->Output")
	testweightstwo:= []float64{0.065,0.156,0.096}
	hiddenToOutput.InitialiseWithWeights(testweightstwo)
	hiddenToOutput.SetActivateSigmoid()

	hiddenToOutputtrainer:= NeuralNet.NewBackPropagation(NeuralNet.SigmoidDerivative)
	inputToHiddentrainer:= NeuralNet.NewBackPropagation(NeuralNet.SigmoidDerivative)

	errorAccm:=[]float64{}

	fmt.Println("**************TS")
	p:= NeuralNet.AddBiasToInputs(data)
	hiddenoutputs:=inputToHiddenLayer.Activate(p, os.Stdout)
    hiddenoutputs=NeuralNet.AddBiasToInputs(hiddenoutputs)
	outputs:= hiddenToOutput.Activate(hiddenoutputs, os.Stdout)
	localError:= NeuralNet.SquaredErrorSum(outputs,target)
	fmt.Printf("LocalError: %f\n" ,localError)
	errorAccm=append(errorAccm,localError)
	wi, _:=hiddenToOutputtrainer.CalculateGradientsOutputLayer(hiddenToOutput,target,os.Stdout)
	
	wo, _:=inputToHiddentrainer.CalculateGradientsHiddenLayer(inputToHiddenLayer,hiddenToOutput,os.Stdout)
	
	hiddenToOutput.VisualiseTransformation(os.Stdout)
	hiddenToOutput.UpdateWeights(wi,0.8)
	inputToHiddenLayer.VisualiseTransformation(os.Stdout)
	inputToHiddenLayer.UpdateWeights(wo,0.8)
	fmt.Println("!!!!!!!!!!!!!!!!TS")
	
}


func TrainingTwo() {

	data := [][]float64{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}
	target := [][]float64{
		{0.01,0.99},
		{0.01,0.99},
		{0.01,0.99},
		{0.01,0.99},
	}

	inputToHiddenLayer:=NeuralNet.NewNamedTransformation(2, true, 2,"Input->Hidden")
	testweightsone:= []float64{0.15,0.2,0.35,0.2,0.3,0.35}
	inputToHiddenLayer.InitialiseWithWeights(testweightsone)
	inputToHiddenLayer.SetActivateSigmoid()
	inputToHiddenLayer.VisualiseWeights(os.Stdout)

	hiddenToOutput:=NeuralNet.NewNamedTransformation(2, true, 2,"Hidden->Output")
	testweightstwo:= []float64{0.4,0.45,0.6,0.5,0.55,0.6}
	hiddenToOutput.InitialiseWithWeights(testweightstwo)
	hiddenToOutput.SetActivateSigmoid()

	hiddenToOutputtrainer:= NeuralNet.NewBackPropagation(NeuralNet.SigmoidDerivative)
	inputToHiddentrainer:= NeuralNet.NewBackPropagation(NeuralNet.SigmoidDerivative)

	
    lasttserror:=make([]float64,4)
    for epoch:=0;epoch<100;epoch++ {
		errorAccm:=[]float64{}
		
		for i, p := range data {
			
			in:= NeuralNet.AddBiasToInputs(p)
			hiddenoutputs:=inputToHiddenLayer.Activate(in, nil)
			hiddenoutputs=NeuralNet.AddBiasToInputs(hiddenoutputs)
			outputs:= hiddenToOutput.Activate(hiddenoutputs, nil)
			localError:= NeuralNet.SquaredErrorSum(outputs,target[i])
			var errchange float64
			if (epoch>0) {
				errchange=localError-float64(lasttserror[i])
			}
			lasttserror[i]=localError
			fmt.Printf("Actual %f %f Target %f Error %f (Change %f)\n" ,outputs[0],outputs[1],target[i],localError,errchange)
			errorAccm=append(errorAccm,localError)

			wi, _:=hiddenToOutputtrainer.CalculateGradientsOutputLayer(hiddenToOutput,target[i],nil)
			wo, _:=inputToHiddentrainer.CalculateGradientsHiddenLayer(inputToHiddenLayer,hiddenToOutput,nil)
			//hiddenToOutput.VisualiseWeights(os.Stdout)
			//inputToHiddenLayer.VisualiseWeights(os.Stdout)
			hiddenToOutput.UpdateWeights(wi,0.8)
			inputToHiddenLayer.UpdateWeights(wo,0.8)
			
		}
			
		epochError:=NeuralNet.MeanError(errorAccm)
		fmt.Printf("EpochError %d: %f\n" ,epoch,epochError)
	}
	
	


}

func TrainingTestTwo() {

	data:=[]float64{0.05,0.1}
	target:=[]float64{0.01,0.99}
	inputToHiddenLayer:=NeuralNet.NewNamedTransformation(2, true, 2,"Input->Hidden")
	testweightsone:= []float64{0.15,0.2,0.35,0.2,0.3,0.35}
	inputToHiddenLayer.InitialiseWithWeights(testweightsone)
	inputToHiddenLayer.SetActivateSigmoid()
	inputToHiddenLayer.VisualiseWeights(os.Stdout)

	hiddenToOutput:=NeuralNet.NewNamedTransformation(2, true, 2,"Hidden->Output")
	testweightstwo:= []float64{0.4,0.45,0.6,0.5,0.55,0.6}
	hiddenToOutput.InitialiseWithWeights(testweightstwo)
	hiddenToOutput.SetActivateSigmoid()

	hiddenToOutputtrainer:= NeuralNet.NewBackPropagation(NeuralNet.SigmoidDerivative)
	inputToHiddentrainer:= NeuralNet.NewBackPropagation(NeuralNet.SigmoidDerivative)


	errorAccm:=[]float64{}

	
	fmt.Println("**************TS")
	p:= NeuralNet.AddBiasToInputs(data)
	hiddenoutputs:=inputToHiddenLayer.Activate(p, os.Stdout)
    hiddenoutputs=NeuralNet.AddBiasToInputs(hiddenoutputs)
	outputs:= hiddenToOutput.Activate(hiddenoutputs, os.Stdout)
	localError:= NeuralNet.SquaredErrorSum(outputs,target)
	fmt.Printf("LocalError: %f\n" ,localError)
	errorAccm=append(errorAccm,localError)
	wi, _:=hiddenToOutputtrainer.CalculateGradientsOutputLayer(hiddenToOutput,target,os.Stdout)
	
	wo, _:=inputToHiddentrainer.CalculateGradientsHiddenLayer(inputToHiddenLayer,hiddenToOutput,os.Stdout)
	
	hiddenToOutput.VisualiseTransformation(os.Stdout)
	hiddenToOutput.UpdateWeights(wi,0.5)
	hiddenToOutput.VisualiseTransformation(os.Stdout)
	inputToHiddenLayer.VisualiseTransformation(os.Stdout)
	inputToHiddenLayer.UpdateWeights(wo,0.5)
	fmt.Println("!!!!!!!!!!!!!!!!TS")
	
	


}


func TrainingThree() {

	data := [][]float64{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}
	target := [][]float64{
		{0.01,0.99},
		{0.01,0.99},
		{0.01,0.99},
		{0.01,0.99},
	}

	inputToHiddenLayer:=NeuralNet.NewNamedTransformation(2, true, 3,"Input->Hidden")

	inputToHiddenLayer.Initialise()
	inputToHiddenLayer.SetActivateSigmoid()
	inputToHiddenLayer.VisualiseWeights(os.Stdout)

	hiddenToOutput:=NeuralNet.NewNamedTransformation(3, true, 2,"Hidden->Output")
	hiddenToOutput.Initialise()
	hiddenToOutput.SetActivateSigmoid()

	hiddenToOutputtrainer:= NeuralNet.NewBackPropagation(NeuralNet.SigmoidDerivative)
	inputToHiddentrainer:= NeuralNet.NewBackPropagation(NeuralNet.SigmoidDerivative)

	
    lasttserror:=make([]float64,4)
    for epoch:=0;epoch<1000;epoch++ {
		errorAccm:=[]float64{}
		
		for i, p := range data {
			
			in:= NeuralNet.AddBiasToInputs(p)
			hiddenoutputs:=inputToHiddenLayer.Activate(in, nil)
			hiddenoutputs=NeuralNet.AddBiasToInputs(hiddenoutputs)
			outputs:= hiddenToOutput.Activate(hiddenoutputs, nil)
			localError:= NeuralNet.SquaredErrorSum(outputs,target[i])
			var errchange float64
			if (epoch>0) {
				errchange=localError-float64(lasttserror[i])
			}
			lasttserror[i]=localError
			fmt.Printf("Actual %f %f Target %f Error %f (Change %f)\n" ,outputs[0],outputs[1],target[i],localError,errchange)
			errorAccm=append(errorAccm,localError)

			wi, _:=hiddenToOutputtrainer.CalculateGradientsOutputLayer(hiddenToOutput,target[i],nil)
			wo, _:=inputToHiddentrainer.CalculateGradientsHiddenLayer(inputToHiddenLayer,hiddenToOutput,nil)

			hiddenToOutput.UpdateWeights(wi,0.5)
			inputToHiddenLayer.UpdateWeights(wo,0.5)
			
		}
			
		epochError:=NeuralNet.MeanError(errorAccm)
		fmt.Printf("EpochError %d: %f\n" ,epoch,epochError)
	}
	
	


}