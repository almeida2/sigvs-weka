package com.fatec.sigvs.controller;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import com.fatec.sigvs.model.IrisRequest;
import com.fatec.sigvs.service.PredictionService;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/iris")
public class IrisController {

    private static final Logger logger = LogManager.getLogger(IrisController.class);
    private final PredictionService predictionService;

    public IrisController(PredictionService predictionService) {
        this.predictionService = predictionService;
    }

    @PostMapping("/classify")
    public ResponseEntity<?> classifyFlower(@RequestBody IrisRequest request) throws Exception {
        logger.info("Received classification request - SepalLength: {}, SepalWidth: {}, PetalLength: {}, PetalWidth: {}",
                request.getSepalLength(), request.getSepalWidth(), request.getPetalLength(), request.getPetalWidth());

        // Validação de campos ausentes/nulos
        if (request.getSepalLength() == null || request.getSepalWidth() == null ||
            request.getPetalLength() == null || request.getPetalWidth() == null) {
            logger.warn("Validation failed: missing required attributes in request");
            return ResponseEntity.badRequest().body(Map.of("error", "Todos os atributos (sepalLength, sepalWidth, petalLength, petalWidth) são obrigatórios."));
        }

        // Validação de valores inválidos (menores ou iguais a zero)
        if (request.getSepalLength() <= 0 || request.getSepalWidth() <= 0 ||
            request.getPetalLength() <= 0 || request.getPetalWidth() <= 0) {
            logger.warn("Validation failed: attributes must be greater than zero");
            return ResponseEntity.badRequest().body(Map.of("error", "Todos os atributos devem ser maiores que zero."));
        }

        try {
            // 1. Criar a estrutura do Dataset e Instância
            Instances dataset = createDatasetStructure();
            DenseInstance instance = new DenseInstance(5);
            instance.setValue(dataset.attribute(0), request.getSepalLength());
            instance.setValue(dataset.attribute(1), request.getSepalWidth());
            instance.setValue(dataset.attribute(2), request.getPetalLength());
            instance.setValue(dataset.attribute(3), request.getPetalWidth());
            instance.setDataset(dataset);

            // 2. Obter a distribuição de probabilidades
            double[] distribution = predictionService.getClassifier().distributionForInstance(instance);

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
            IrisResponse response = new IrisResponse(predictedClass, maxConfidence);

            logger.info("Successfully classified. Class: {}, Confidence: {}%", 
                    response.getPredictedClass(), response.getConfidence());

            return ResponseEntity.ok(response);

        } catch (Exception e) {
            logger.error("Error during iris classification process", e);
            throw e;
        }
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
