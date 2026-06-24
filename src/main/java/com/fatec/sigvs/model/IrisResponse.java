package com.fatec.sigvs.model;

public class IrisResponse {
    private String predictedClass;
    private double confidence;

    public IrisResponse(String predictedClass, double confidence) {
        this.predictedClass = predictedClass;
        // Transforma o valor decimal em porcentagem formatada (ex: 0.985 -> 98.5)
        this.confidence = Math.round(confidence * 1000.0) / 10.0;
    }

    // Getters e Setters
    public String getPredictedClass() {
        return predictedClass;
    }

    public void setPredictedClass(String predictedClass) {
        this.predictedClass = predictedClass;
    }

    public double getConfidence() {
        return confidence;
    }

    public void setConfidence(double confidence) {
        this.confidence = confidence;
    }
}
