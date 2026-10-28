package com.fatec.sigvs.model;

/**
 * Classe que representa a requisição de classificação de flores Iris
 * mapeado pelo arquivo json recebido na classe IrisController
 * 
 * @see IrisController#classifyFlower(IrisRequest)
 */
public class IrisRequest {
    private Double sepalLength;
    private Double sepalWidth;
    private Double petalLength;
    private Double petalWidth;

    // Getters e Setters
    public Double getSepalLength() {
        return sepalLength;
    }

    public void setSepalLength(Double sepalLength) {
        this.sepalLength = sepalLength;
    }

    public Double getSepalWidth() {
        return sepalWidth;
    }

    public void setSepalWidth(Double sepalWidth) {
        this.sepalWidth = sepalWidth;
    }

    public Double getPetalLength() {
        return petalLength;
    }

    public void setPetalLength(Double petalLength) {
        this.petalLength = petalLength;
    }

    public Double getPetalWidth() {
        return petalWidth;
    }

    public void setPetalWidth(Double petalWidth) {
        this.petalWidth = petalWidth;
    }
}
