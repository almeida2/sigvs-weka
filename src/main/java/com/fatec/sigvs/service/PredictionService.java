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

    public double predict(double[] featureValues) throws Exception {
        // Create an instance bound to the dataset structure
        DenseInstance instance = new DenseInstance(1.0, featureValues);
        instance.setDataset(datasetStructure);

        // Perform prediction (returns class index or numerical value)
        return classifier.classifyInstance(instance);
    }

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
     * Auxiliar para definir a estrutura do dataset de acordo com as especificações do Weka.
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
