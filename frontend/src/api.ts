import { PredictResponse } from "./types";

const API_BASE = "http://localhost:8000";

export async function classifyImage(file: File): Promise<PredictResponse> {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${API_BASE}/predict`, { method: "POST", body: form });
  if (!res.ok) {
    const detail = await res.json().catch(() => null);
    throw new Error(detail?.detail ?? `Request failed (${res.status})`);
  }
  return res.json();
}