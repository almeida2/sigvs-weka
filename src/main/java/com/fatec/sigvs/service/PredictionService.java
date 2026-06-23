package com.fatec.sigvs.service;

import org.springframework.stereotype.Service;
import weka.classifiers.Classifier;
import weka.core.DenseInstance;
import weka.core.Instances;
import jakarta.annotation.PostConstruct;
import java.io.InputStream;

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

        // 2. Define your dataset schema (features) programmatically
        // Or load a template .arff/.csv header file to configure attributes
        // this.datasetStructure = ... (must match the model's exact expected
        // attributes)
    }

    public double predict(double[] featureValues) throws Exception {
        // Create an instance bound to the dataset structure
        DenseInstance instance = new DenseInstance(1.0, featureValues);
        instance.setDataset(datasetStructure);

        // Perform prediction (returns class index or numerical value)
        return classifier.classifyInstance(instance);
    }

    public Classifier getClassifier() {
        return classifier;
    }
}
