export interface Prediction {
  aircraft: string;
  confidence: number;
}

export interface PredictResponse {
  aircraft: string;
  confidence: number;
  predictions: Prediction[];
}