import { PredictResponse } from "../types";

export default function PredictionResult({ data }: { data: PredictResponse }) {
  return (
    <div className="bg-navy-700 rounded-xl p-6 mt-6">
      <p className="text-slate-400 text-sm mb-1">Top match</p>
      <h2 className="text-2xl font-bold text-radar">{data.aircraft}</h2>
      <p className="text-slate-300 mb-4">{(data.confidence * 100).toFixed(1)}% confidence</p>

      <div className="space-y-2">
        {data.predictions.map((p) => (
          <div key={p.aircraft} className="flex items-center gap-3">
            <span className="w-32 text-sm text-slate-300">{p.aircraft}</span>
            <div className="flex-1 h-2 bg-navy-900 rounded-full overflow-hidden">
              <div
                className="h-full bg-radar"
                style={{ width: `${p.confidence * 100}%` }}
              />
            </div>
            <span className="w-14 text-right text-sm text-slate-400">
              {(p.confidence * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}