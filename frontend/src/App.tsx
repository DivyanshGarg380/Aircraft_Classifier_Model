import { useState } from "react";
import ImageUpload from "./components/ImageUpload";
import PredictionResult from "./components/PredictionResult";
import { classifyImage } from "./api";
import { PredictResponse } from "./types";

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleClassify() {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await classifyImage(file);
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-navy-900 text-white flex items-center justify-center p-6">
      <div className="w-full max-w-md">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold tracking-tight">✈️ Aircraft Classifier</h1>
          <p className="text-slate-400 mt-1">Upload a photo, get the aircraft type</p>
        </div>

        <ImageUpload onFileSelected={setFile} />

        <button
          onClick={handleClassify}
          disabled={!file || loading}
          className="w-full mt-4 py-3 rounded-lg font-semibold bg-radar text-navy-900
                     disabled:opacity-40 disabled:cursor-not-allowed hover:opacity-90 transition"
        >
          {loading ? "Classifying..." : "Classify Aircraft"}
        </button>

        {error && (
          <div className="mt-4 p-3 rounded-lg bg-red-900/40 border border-red-700 text-red-300 text-sm">
            {error}
          </div>
        )}

        {result && <PredictionResult data={result} />}
      </div>
    </div>
  );
}