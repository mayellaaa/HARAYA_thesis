export type Speaker = "User A" | "User B";

export interface Message {
  speaker: Speaker;
  text: string;
  role?: "sender" | "receiver" | "participant" | "unknown";
}

export interface IndicatorEvidence {
  id: string;
  label: string;
  present: boolean;
  source: string;
  evidence: string[];
  similarity: number | null;
}

export interface CategoryScore {
  label: string;
  score: number;
  max_score: number;
  decision: "Insufficient evidence" | "Possible" | "Classified";
  matched_indicators: string[];
  legal_reference: string;
}

export interface AnalyzeResponse {
  conversation_text: string;
  prediction: string;
  confidence: number;
  rule_score: number;
  rule_max_score: number;
  detected: boolean;
  language: string;
  source: string;
  legal_explanation: string;
  legal_indicators: IndicatorEvidence[];
  category_scores: CategoryScore[];
  processed_messages: Array<{
    speaker: string;
    original: string;
    cleaned: string;
    normalized: string;
    tokens: string[];
    language: string;
  }>;
  mbert: {
    encoder_loaded: boolean;
    classifier_loaded: boolean;
    role: string;
    model_id: string;
    classifier_id: string | null;
    encoder_error: string | null;
    classifier_error: string | null;
    distribution: Array<{ label: string; score: number }>;
  };
  model: string;
  disclaimer: string;
}

export interface SampleConversation {
  name: string;
  messages: Message[];
}
