import { Shield, CheckCircle2, AlertTriangle, Layers } from "lucide-react";
import { Badge } from "../ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";

interface RobustnessScenario {
  name: string;
  description: string;
  missingFeatures: number;
  availableFeatures: number;
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  fpr: number;
  cm: {
    tp: number;
    fp: number;
    tn: number;
    fn: number;
  };
}

const ROBUSTNESS_DATA: RobustnessScenario[] = [
  {
    name: "Full Feature Baseline",
    description: "All 30 canonical features present without missing signals",
    missingFeatures: 0,
    availableFeatures: 30,
    accuracy: 97.29,
    precision: 97.24,
    recall: 98.02,
    f1Score: 97.63,
    fpr: 3.68,
    cm: { tp: 1235, fp: 35, tn: 916, fn: 25 },
  },
  {
    name: "Missing Webpage / HTML Signals",
    description: "Website offline or blocked; 10 DOM features safely imputed by KNN",
    missingFeatures: 10,
    availableFeatures: 20,
    accuracy: 92.09,
    precision: 90.88,
    recall: 95.71,
    f1Score: 93.24,
    fpr: 12.72,
    cm: { tp: 1206, fp: 121, tn: 830, fn: 54 },
  },
  {
    name: "Missing External / WHOIS APIs",
    description: "Third-party APIs unreachable; 6 external features safely imputed by KNN",
    missingFeatures: 6,
    availableFeatures: 24,
    accuracy: 93.44,
    precision: 93.59,
    recall: 95.00,
    f1Score: 94.29,
    fpr: 8.62,
    cm: { tp: 1197, fp: 82, tn: 869, fn: 63 },
  },
  {
    name: "Lexical-Only (URL Syntax Alone)",
    description: "Zero network/web access; only 14 lexical URL signals available",
    missingFeatures: 16,
    availableFeatures: 14,
    accuracy: 89.15,
    precision: 89.35,
    recall: 91.90,
    f1Score: 90.61,
    fpr: 14.51,
    cm: { tp: 1158, fp: 138, tn: 813, fn: 102 },
  },
];

function RobustnessTable() {
  return (
    <Card className="mt-8 overflow-hidden">
      <CardHeader>
        <div className="flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Shield className="h-5 w-5 text-teal-600" />
              Missing-Feature Robustness Evaluation
            </CardTitle>
            <p className="mt-1 text-xs text-slate-500">
              Evaluated on 2,211 test samples under 4 simulated missing-signal conditions using KNN imputation.
            </p>
          </div>
          <Badge className="w-fit border-teal-200 bg-teal-50 text-xs text-teal-800">
            Empirical Test Set
          </Badge>
        </div>
      </CardHeader>
      <CardContent>
        <div className="overflow-auto">
          <table className="w-full min-w-[780px] text-left text-sm">
            <thead className="bg-slate-50 text-xs uppercase text-slate-500">
              <tr>
                <th className="px-4 py-3">Scenario</th>
                <th className="px-4 py-3">Missing Signals</th>
                <th className="px-4 py-3">Accuracy</th>
                <th className="px-4 py-3">Precision</th>
                <th className="px-4 py-3">Recall</th>
                <th className="px-4 py-3">F1-Score</th>
                <th className="px-4 py-3">False Positive Rate (FPR)</th>
              </tr>
            </thead>
            <tbody>
              {ROBUSTNESS_DATA.map((row, idx) => (
                <tr
                  key={row.name}
                  className={`border-t border-slate-100 ${
                    idx === 0 ? "bg-teal-50/40 font-medium" : "hover:bg-slate-50"
                  }`}
                >
                  <td className="px-4 py-3.5">
                    <div>
                      <p className="font-semibold text-slate-900">{row.name}</p>
                      <p className="text-xs text-slate-500">{row.description}</p>
                    </div>
                  </td>
                  <td className="px-4 py-3.5">
                    <span className="rounded bg-slate-100 px-2 py-0.5 text-xs text-slate-700">
                      {row.missingFeatures} / 30 missing
                    </span>
                  </td>
                  <td className="px-4 py-3.5 font-medium">{row.accuracy.toFixed(2)}%</td>
                  <td className="px-4 py-3.5">{row.precision.toFixed(2)}%</td>
                  <td className="px-4 py-3.5">{row.recall.toFixed(2)}%</td>
                  <td className="px-4 py-3.5">
                    <Badge
                      className={`text-xs ${
                        row.f1Score >= 95
                          ? "border-teal-200 bg-teal-50 text-teal-800"
                          : "border-slate-200 bg-slate-50 text-slate-700"
                      }`}
                    >
                      {row.f1Score.toFixed(2)}%
                    </Badge>
                  </td>
                  <td className="px-4 py-3.5">
                    <span
                      className={`font-semibold ${
                        row.fpr <= 5
                          ? "text-teal-700"
                          : row.fpr <= 10
                            ? "text-amber-700"
                            : "text-slate-700"
                      }`}
                    >
                      {row.fpr.toFixed(2)}%
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="mt-6 grid gap-3 rounded-lg border border-slate-200 bg-slate-50/80 p-4 text-xs text-slate-700 sm:grid-cols-3">
          <div className="flex items-start gap-2">
            <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-teal-600" />
            <div>
              <p className="font-semibold text-slate-900">Zero False Penalization</p>
              <p className="mt-0.5 text-slate-600">
                Websites are never automatically classified as phishing merely because access fails or network times out.
              </p>
            </div>
          </div>
          <div className="flex items-start gap-2">
            <Layers className="mt-0.5 h-4 w-4 shrink-0 text-teal-600" />
            <div>
              <p className="font-semibold text-slate-900">KNN Imputation Resilience</p>
              <p className="mt-0.5 text-slate-600">
                Preserves 93.24% F1 even when all 10 webpage/DOM features are missing, and 90.61% F1 in lexical-only mode.
              </p>
            </div>
          </div>
          <div className="flex items-start gap-2">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-600" />
            <div>
              <p className="font-semibold text-slate-900">Low False Positive Rate</p>
              <p className="mt-0.5 text-slate-600">
                FPR remains strictly controlled (3.68% baseline, 12.72% without DOM access), avoiding false blockage of legitimate links.
              </p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export { RobustnessTable };
