package com.fatec.sigvs.service;

import org.springframework.stereotype.Service;
import com.fatec.sigvs.model.IrisRequest;
import com.fatec.sigvs.model.IrisResponse;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import jakarta.annotation.PostConstruct;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

@Service
public class PredictionService {

    private Classifier classifier;
    private Instances datasetStructure; // Defines features and labels

    /**
     * Método de inicialização que carrega o modelo de classificação pré-treinado
     * é executado logo apos a injeção de dependencias do bean ter sido executadada
     * 
     * @throws Exception - dispara uma exception caso o modelo nao seja encontrado
     */
    @PostConstruct
    public void init() throws Exception {
        // 1. Load the pre-trained Weka model from resources
        try (InputStream is = getClass().getResourceAsStream("/classifier.model")) {
            if (is == null)
                throw new IllegalArgumentException("Model file not found!");
            this.classifier = (Classifier) weka.core.SerializationHelper.read(is);
        }

        // Initialize datasetStructure
        this.datasetStructure = createDatasetStructure();
    }

    /**
     * Recebe um array de valores de entrada e retorna a classe prevista para a
     * florIris
     * 
     * @param featureValues - array de valores de entrada
     * @return - valor da classe prevista
     * @throws Exception - dispara uma exceção caso a predição falhe
     */
    public double predict(double[] featureValues) throws Exception {
        // Cria uma instancia com os valores de entrada
        // calcula a distancia por similaridade de coseno
        DenseInstance instance = new DenseInstance(1.0, featureValues);
        // Define a estrutura do dataset
        instance.setDataset(datasetStructure);
        // Realiza a predição
        return classifier.classifyInstance(instance);
    }

    /**
     * Recebe um objeto IrisRequest e retorna um objeto IrisResponse com a classe
     * prevista
     * 
     * @param request - objeto IrisRequest com os valores de entrada
     * @return - objeto IrisResponse com a classe prevista
     * @throws Exception - dispara uma exceção caso a predição falhe
     */
    public IrisResponse classify(IrisRequest request) throws Exception {
        // 1. Criar a estrutura do Dataset e Instância
        Instances dataset = createDatasetStructure();
        DenseInstance instance = new DenseInstance(5);
        instance.setValue(dataset.attribute(0), request.getSepalLength());
        instance.setValue(dataset.attribute(1), request.getSepalWidth());
        instance.setValue(dataset.attribute(2), request.getPetalLength());
        instance.setValue(dataset.attribute(3), request.getPetalWidth());
        instance.setDataset(dataset);

        // 2. Obter a distribuição de probabilidades
        double[] distribution = classifier.distributionForInstance(instance);

        // 3. Encontrar o índice com a maior probabilidade
        int maxIndex = 0;
        double maxConfidence = 0.0;
        for (int i = 0; i < distribution.length; i++) {
            if (distribution[i] > maxConfidence) {
                maxConfidence = distribution[i];
                maxIndex = i;
            }
        }

        // 4. Obter a espécie correspondente
        String predictedClass = dataset.classAttribute().value(maxIndex);
        return new IrisResponse(predictedClass, maxConfidence);
    }

    public Classifier getClassifier() {
        return classifier;
    }

    /**
     * Auxiliar para definir a estrutura do dataset de acordo com as especificações
     * do Weka.
     */
    private Instances createDatasetStructure() {
        ArrayList<Attribute> attributes = new ArrayList<>();
        attributes.add(new Attribute("sepallength"));
        attributes.add(new Attribute("sepalwidth"));
        attributes.add(new Attribute("petallength"));
        attributes.add(new Attribute("petalwidth"));

        ArrayList<String> classValues = new ArrayList<>(List.of("Iris-setosa", "Iris-versicolor", "Iris-virginica"));
        attributes.add(new Attribute("class", classValues));

        Instances dataset = new Instances("IrisDataset", attributes, 0);
        dataset.setClassIndex(dataset.numAttributes() - 1);
        return dataset;
    }
}
