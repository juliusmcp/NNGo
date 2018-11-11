package NeuralNet

import (
	"sync"
)

type TransformationList struct {
	head, tail 				*Transformation
}

type NeuralNet struct {
	TransformationCount 	int
	transformations 		*TransformationList
	lock  					sync.RWMutex
}

func NewNetwork() *NeuralNet {
	return &NeuralNet{
		transformations: new(TransformationList),
	}
}

func (net *NeuralNet) FirstLayer() *Transformation {
	return net.transformations.head
}

func (net *NeuralNet) LastLayer() *Transformation {
	return net.transformations.tail
}

/*func (net *NeuralNet) FindLayer(index int) *Layer {
	/(currentLayer:=net.layers.head
	currentIndex:=currentLayer.Index
	while(currentLayer!=nil) {
		currentLayer=currentLayer.nextLayer
		currentIndex=currentLayer.Index
	}
	return currentLayer
}*/
func (net *NeuralNet) AppendTransformation(transformation *Transformation) {
	net.lock.Lock()
	defer net.lock.Unlock()
	net.TransformationCount++
	if net.transformations.head==nil {
		transformation.index=0
		net.transformations.head=transformation
		net.transformations.tail=transformation
	} else {
		transformation.index=net.transformations.tail.index+1
		transformation.prevLayer=net.transformations.tail
		net.transformations.tail=transformation
		transformation.prevLayer.nextLayer=transformation
	}
}


func (layer *Transformation) Next() *Transformation {
	return layer.nextLayer
}

func (layer *Transformation) Prev() *Transformation {
	return layer.prevLayer
}


