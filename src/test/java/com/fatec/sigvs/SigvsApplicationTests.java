package com.fatec.sigvs;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import com.fatec.sigvs.service.PredictionService;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import java.util.ArrayList;
import java.util.List;

@SpringBootTest
class SigvsApplicationTests {

    @Autowired
    private PredictionService predictionService;

    @Test
    void contextLoads() {
    }

    @Test
    void testClassifier() throws Exception {
        System.out.println("=================================================");
        System.out.println("CLASSIFIER CLASS: " + predictionService.getClassifier().getClass().getName());
        System.out.println("CLASSIFIER DETAILS:\n" + predictionService.getClassifier());
        System.out.println("=================================================");

        // Test with different attributes
        ArrayList<Attribute> attributes = new ArrayList<>();
        attributes.add(new Attribute("sepallength"));
        attributes.add(new Attribute("sepalwidth"));
        attributes.add(new Attribute("petallength"));
        attributes.add(new Attribute("petalwidth"));
        attributes.add(new Attribute("class", new ArrayList<>(List.of("Iris-setosa", "Iris-versicolor", "Iris-virginica"))));

        Instances dataset = new Instances("IrisDataset", attributes, 0);
        dataset.setClassIndex(dataset.numAttributes() - 1);

        double[][] testInputs = {
            {5.1, 3.5, 1.4, 0.2}, // Setosa
            {6.0, 2.9, 4.5, 1.5}, // Versicolor
            {6.5, 3.0, 5.8, 2.2}  // Virginica
        };

        for (double[] inputs : testInputs) {
            DenseInstance instance = new DenseInstance(5);
            instance.setValue(attributes.get(0), inputs[0]);
            instance.setValue(attributes.get(1), inputs[1]);
            instance.setValue(attributes.get(2), inputs[2]);
            instance.setValue(attributes.get(3), inputs[3]);
            instance.setDataset(dataset);

            double[] distribution = predictionService.getClassifier().distributionForInstance(instance);
            System.out.printf("Input: %s -> Distribution: ", java.util.Arrays.toString(inputs));
            for (int i = 0; i < distribution.length; i++) {
                System.out.printf("%s: %.2f%% ", dataset.classAttribute().value(i), distribution[i] * 100);
            }
            System.out.println();
        }
        System.out.println("=================================================");
    }
}
