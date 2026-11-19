interface StatusItem {
  meta: string;
  ready: boolean;
  status: string;
}

interface AppStatus {
  gemini: StatusItem;
  model: StatusItem;
  modelComparison: StatusItem;
  mongo: StatusItem;
  output: StatusItem;
  preprocessor: StatusItem;
  repo: {
    name: string;
    url: string;
  };
  sampleData: StatusItem;
}

interface UrlFeature {
  name: string;
  signal: "Suspicious" | "Neutral" | "Normal";
  value: number;
}

interface FeatureContribution {
  category?: string;
  direction: "Phishing" | "Legitimate" | string;
  display_value?: string;
  displayValue?: string;
  explanation: string;
  feature: string;
  feature_name?: string;
  featureName?: string;
  importance: number;
  is_unavailable?: boolean;
  isUnavailable?: boolean;
  raw_value?: number | null;
  rawValue?: number | null;
  shap_value?: number;
  shapValue?: number;
  supports_prediction?: boolean;
  supportsPrediction?: boolean;
}

interface UrlPrediction {
  availableFeaturesCount?: number;
  confidence: number;
  createdAt?: string;
  dnsStatus?: string;
  explainability?: {
    base_value?: number;
    method?: string;
  };
  features?: UrlFeature[] | Record<string, number | null>;
  finalUrl?: string;
  historyId?: string;
  hostname?: string;
  htmlStatus?: string;
  label: "Legitimate" | "Phishing" | number;
  missingFeaturesCount?: number;
  prediction: "Legitimate" | "Phishing" | number;
  probabilities?: {
    phishing: number;
    legitimate: number;
  };
  risk_level?: string;
  signals?: {
    neutral: number;
    normal: number;
    suspicious: number;
  };
  statusClass?: "good" | "danger";
  summary?: string;
  suspiciousFeatures?: string[];
  top_contributions?: FeatureContribution[];
  topContributions?: FeatureContribution[];
  unavailable_warning?: string | null;
  unavailableWarning?: string | null;
  url: string;
}

interface CsvPrediction {
  columns: number;
  fileName: string;
  legitimate: number;
  outputMeta: string;
  phishing: number;
  preview: Array<Record<string, string | number>>;
  rows: number;
}

interface PredictionHistoryResponse {
  items: UrlPrediction[];
  requiresAuth?: boolean;
}

interface AuthUser {
  createdAt: string;
  email: string;
  id: string;
  name: string;
}

interface AuthSession {
  expiresAt: string;
  token: string;
  user: AuthUser;
}

interface AuthCredentials {
  email: string;
  name?: string;
  password: string;
}

interface ModelComparisonRow {
  bestParams: Record<string, string | number | boolean | null>;
  error: string;
  isBest: boolean;
  modelName: string;
  status: "trained" | "failed" | "unavailable";
  testAccuracy: number;
  testF1: number;
  testPrecision: number;
  testRecall: number;
  trainAccuracy: number;
  trainF1: number;
  trainPrecision: number;
  trainRecall: number;
}

interface ModelComparisonReport {
  bestModelName: string;
  bestModelScore: number;
  generatedAt: string;
  message?: string;
  models: ModelComparisonRow[];
  ready: boolean;
  reportMeta?: string;
}

export type {
  AppStatus,
  AuthCredentials,
  AuthSession,
  AuthUser,
  CsvPrediction,
  FeatureContribution,
  ModelComparisonReport,
  ModelComparisonRow,
  PredictionHistoryResponse,
  UrlFeature,
  UrlPrediction,
};
