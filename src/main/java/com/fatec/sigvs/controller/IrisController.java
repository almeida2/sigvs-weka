package com.fatec.sigvs.controller;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import com.fatec.sigvs.model.IrisRequest;
import com.fatec.sigvs.model.IrisResponse;
import com.fatec.sigvs.service.PredictionService;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

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
            // Obter a resposta delegando ao serviço de classificação
            IrisResponse response = predictionService.classify(request);

            logger.info("Successfully classified. Class: {}, Confidence: {}%", 
                    response.getPredictedClass(), response.getConfidence());

            return ResponseEntity.ok(response);

        } catch (Exception e) {
            logger.error("Error during iris classification process", e);
            throw e;
        }
    }
}
