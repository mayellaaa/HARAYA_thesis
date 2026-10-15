import type { AnalyzeResponse, Message } from "./types";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

export async function analyzeConversation(
  messages: Message[],
  source = "Manual Input",
): Promise<AnalyzeResponse> {
  const response = await fetch(`${API}/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages, source }),
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || "Analysis failed.");
  }
  return response.json();
}

export async function transcribeImage(dataUrl: string): Promise<{
  bubbles: Array<{ speaker: Message["speaker"]; text: string; side: string }>;
  error?: string | null;
}> {
  const response = await fetch(`${API}/ocr`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ data_url: dataUrl }),
  });
  if (!response.ok) {
    throw new Error("Image transcription failed.");
  }
  return response.json();
}
