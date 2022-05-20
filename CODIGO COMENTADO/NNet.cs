using System.Collections;
using System.Collections.Generic;
using UnityEngine;

// DLL's que nos que incluyen los algoritmos de red neuronal PERCEPTRON basados en C#
using MathNet.Numerics.LinearAlgebra;
using System;

using Random = UnityEngine.Random; // Librería para cálculos aleatorios

public class NNet : MonoBehaviour
{
    public Matrix<float> inputLayer = Matrix<float>.Build.Dense(1, 3); // Definición de vector de entradas

    public List<Matrix<float>> hiddenLayers = new List<Matrix<float>>(); // Definición de capas ocultas como matriz

    public Matrix<float> outputLayer = Matrix<float>.Build.Dense(1, 2);// Definición de vector de salidas

    public List<Matrix<float>> weights = new List<Matrix<float>>(); // Definición de capas ocultas como matriz

    public List<float> biases = new List<float>(); // Definición de capas ocultas como matriz

    public float fitness; // variable ligada al cálculo de redes neuronales óptimas, recibe valores de Genetic Manager

// Método principal donde se reciben el número de capas ocultas y el número de neuronas para entrenar.
    public void Initialise(int hiddenLayerCount, int hiddenNeuronCount)
    {

        // limpieza de vectores y matrices para evitar ambiguedades en los cáculos
        inputLayer.Clear();
        hiddenLayers.Clear();
        outputLayer.Clear();
        weights.Clear();
        biases.Clear();

        for (int i = 0; i < hiddenLayerCount + 1; i++)
        {

            Matrix<float> f = Matrix<float>.Build.Dense(1, hiddenNeuronCount);

            hiddenLayers.Add(f);

            biases.Add(Random.Range(-1f, 1f));

            //WEIGHTS
            if (i == 0)
            {
                Matrix<float> inputToH1 = Matrix<float>.Build.Dense(3, hiddenNeuronCount);
                weights.Add(inputToH1);
            }

            Matrix<float> HiddenToHidden = Matrix<float>.Build.Dense(hiddenNeuronCount, hiddenNeuronCount);
            weights.Add(HiddenToHidden);

        }

        // Aleatoriedad en los valores de los pesos y las BIAS EN UN RANGO DE [-1, 1]
        Matrix<float> OutputWeight = Matrix<float>.Build.Dense(hiddenNeuronCount, 2);
        weights.Add(OutputWeight);
        biases.Add(Random.Range(-1f, 1f));

        RandomiseWeights();

    }

// Método alternatico a Inicialice que se ocupa cuando se están haciendo las comparaciones entre redes neuronales
    public NNet InitialiseCopy(int hiddenLayerCount, int hiddenNeuronCount)
    {
        NNet n = new NNet();

        List<Matrix<float>> newWeights = new List<Matrix<float>>();

        for (int i = 0; i < this.weights.Count; i++)
        {
            Matrix<float> currentWeight = Matrix<float>.Build.Dense(weights[i].RowCount, weights[i].ColumnCount);

            for (int x = 0; x < currentWeight.RowCount; x++)
            {
                for (int y = 0; y < currentWeight.ColumnCount; y++)
                {
                    currentWeight[x, y] = weights[i][x, y];
                }
            }

            newWeights.Add(currentWeight);
        }

        List<float> newBiases = new List<float>();

        newBiases.AddRange(biases);

        n.weights = newWeights;
        n.biases = newBiases;

        n.InitialiseHidden(hiddenLayerCount, hiddenNeuronCount);

        // Se retorna un cierto número de redes neuronales y posteriormente se observa cuáles son las que tuvieron mejor rendimiento
        return n;
    }

// Método donde entrenan las Capas ocultas. Se recibe un cierto númeto de capas y un cierto númeo de neuronas para entrenar
    public void InitialiseHidden(int hiddenLayerCount, int hiddenNeuronCount)
    {
        inputLayer.Clear();
        hiddenLayers.Clear();
        outputLayer.Clear();

        for (int i = 0; i < hiddenLayerCount + 1; i++)
        {
            Matrix<float> newHiddenLayer = Matrix<float>.Build.Dense(1, hiddenNeuronCount);
            hiddenLayers.Add(newHiddenLayer);
        }

    }

// Método para la aleatorización de los pesos. Tomando valores de [-1, 1]
    public void RandomiseWeights()
    {

        for (int i = 0; i < weights.Count; i++)
        {

            for (int x = 0; x < weights[i].RowCount; x++)
            {

                for (int y = 0; y < weights[i].ColumnCount; y++)
                {

                    weights[i][x, y] = Random.Range(-1f, 1f);

                }

            }

        }

    }

// Método para el entrenamiento individual de una red neuronal. Aquí se ingresan los parámetro de los 3 sensores
    public (float, float) RunNetwork(float a, float b, float c)
    {
        // Distribución de valores por celda en el vector de entradas
        inputLayer[0, 0] = a;
        inputLayer[0, 1] = b;
        inputLayer[0, 2] = c;

// Utilización de ajuste con la función de Tangente Hoiperbbólica
        inputLayer = inputLayer.PointwiseTanh();

        hiddenLayers[0] = ((inputLayer * weights[0]) + biases[0]).PointwiseTanh();

        for (int i = 1; i < hiddenLayers.Count; i++)
        {
            hiddenLayers[i] = ((hiddenLayers[i - 1] * weights[i]) + biases[i]).PointwiseTanh();
        }

        outputLayer = ((hiddenLayers[hiddenLayers.Count - 1] * weights[weights.Count - 1]) + biases[biases.Count - 1]).PointwiseTanh();

        //First output is acceleration and second output is steering
        return (Sigmoid(outputLayer[0, 0]), (float)Math.Tanh(outputLayer[0, 1]));
    }

// Utilización de ajuste con la función de Sigmoid
    private float Sigmoid(float s)
    {
        return (1 / (1 + Mathf.Exp(-s)));
    }

}
