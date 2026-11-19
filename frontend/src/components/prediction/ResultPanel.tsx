import {
  BrainCircuit,
  Database,
  Download,
  FileSpreadsheet,
  Globe2,
  Grid3X3,
  Info,
  LineChart,
  LockKeyhole,
  ShieldAlert,
  ShieldCheck,
} from "lucide-react";
import {
  Bar,
  BarChart,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { MiniStatus } from "../common/MiniStatus";
import { Badge } from "../ui/badge";
import { Button } from "../ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import { Progress } from "../ui/progress";
import { apiUrl } from "../../lib/api";
import { buildCsvPreviewColumns, formatConfidence } from "../../lib/format";
import type { ActiveResult } from "../../hooks/usePredictionWorkspace";
import type { CsvPrediction, UrlPrediction } from "../../lib/types";

interface ResultPanelProps {
  activeResult: ActiveResult;
  csvResult: CsvPrediction | null;
  urlResult: UrlPrediction | null;
}

function ResultPanel({ activeResult, csvResult, urlResult }: ResultPanelProps) {
  const isLegitimate =
    urlResult?.prediction === "Legitimate" ||
    urlResult?.label === "Legitimate" ||
    urlResult?.label === 1 ||
    urlResult?.prediction === 1;
  const resultTone = isLegitimate ? "text-teal-600" : "text-red-600";
  const csvChartData = csvResult
    ? [
        { fill: "#ef4444", name: "Phishing", value: csvResult.phishing },
        { fill: "#14b8a6", name: "Legitimate", value: csvResult.legitimate },
      ]
    : [];
  const csvTotal = csvResult ? Math.max(csvResult.phishing + csvResult.legitimate, 1) : 1;
  const csvPhishingPercent = csvResult ? Math.round((csvResult.phishing / csvTotal) * 100) : 0;
  const csvLegitimatePercent = csvResult ? 100 - csvPhishingPercent : 0;
  const previewColumns = csvResult?.preview[0] ? buildCsvPreviewColumns(Object.keys(csvResult.preview[0])) : [];

  return (
    <Card className="overflow-hidden shadow-md shadow-slate-200/70">
      <CardHeader>
        <CardTitle>Live result</CardTitle>
      </CardHeader>
      <CardContent>
        {activeResult === "url" && urlResult ? (
          <UrlResult resultTone={resultTone} urlResult={urlResult} />
        ) : activeResult === "csv" && csvResult ? (
          <CsvResult
            csvChartData={csvChartData}
            csvLegitimatePercent={csvLegitimatePercent}
            csvPhishingPercent={csvPhishingPercent}
            csvResult={csvResult}
            previewColumns={previewColumns}
          />
        ) : (
          <EmptyResult />
        )}
      </CardContent>
    </Card>
  );
}

function UrlResult({ resultTone, urlResult }: { resultTone: string; urlResult: UrlPrediction }) {
  const displayLabel =
    typeof urlResult.prediction === "string"
      ? urlResult.prediction
      : (typeof urlResult.label === "string"
          ? urlResult.label
          : (urlResult.label === 1 || urlResult.prediction === 1 ? "Legitimate" : "Phishing"));
  const confidencePercent = formatConfidence(urlResult.confidence);
  const topContribs = urlResult.topContributions || urlResult.top_contributions || [];
  const unavailableWarning = urlResult.unavailableWarning || urlResult.unavailable_warning;
  const riskLevel = urlResult.risk_level || "Standard";
  const isPhish = displayLabel.toLowerCase() === "phishing";

  return (
    <div className="space-y-5">
      {unavailableWarning && (
        <div className="flex items-start gap-3 rounded-lg border border-amber-200 bg-amber-50 p-4 text-xs text-amber-900 shadow-sm">
          <Info className="mt-0.5 h-4 w-4 shrink-0 text-amber-600" />
          <div>
            <p className="font-semibold text-amber-950">Missing-Feature Robustness Active</p>
            <p className="mt-0.5 leading-relaxed">{unavailableWarning}</p>
          </div>
        </div>
      )}

      <div className="flex flex-col justify-between gap-4 rounded-lg border border-slate-200 bg-slate-50 p-4 sm:flex-row sm:items-center">
        <div>
          <div className="flex items-center gap-2">
            <p className="text-sm text-slate-500">Model prediction</p>
            <Badge
              className={`text-[10px] ${
                isPhish ? "border-red-200 bg-red-100 text-red-800" : "border-teal-200 bg-teal-100 text-teal-800"
              }`}
            >
              {riskLevel} Risk
            </Badge>
          </div>
          <h3 className={`text-3xl font-semibold ${resultTone}`}>{displayLabel}</h3>
        </div>
        <div className="w-full sm:w-56">
          <div className="mb-2 flex justify-between text-sm text-slate-600">
            <span>Confidence</span>
            <span>{confidencePercent}%</span>
          </div>
          <Progress value={confidencePercent} />
        </div>
      </div>

      <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
        <h4 className="mb-1 text-sm font-semibold text-slate-900">Evidence-Grounded AI Explanation</h4>
        <p className="text-sm leading-6 text-slate-600">{urlResult.summary}</p>
      </div>

      {topContribs.length > 0 && (
        <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
          <div className="mb-3 flex items-center justify-between">
            <div>
              <h4 className="flex items-center gap-1.5 text-sm font-semibold text-slate-900">
                <BrainCircuit className="h-4 w-4 text-teal-600" />
                Top 5 Influential Features (SHAP Explainability)
              </h4>
              <p className="text-xs text-slate-500">
                Key mathematical drivers calculated via TreeExplainer for this URL.
              </p>
            </div>
            <Badge className="border-teal-200 bg-teal-50 text-[10px] text-teal-800">
              SHAP AI
            </Badge>
          </div>
          <div className="space-y-2.5">
            {topContribs.map((contrib, idx) => {
              const featName = contrib.feature_name || contrib.featureName || contrib.feature;
              const pushesPhish = contrib.direction === "Phishing";
              const shapVal = contrib.shap_value ?? contrib.shapValue ?? 0;
              const absVal = Math.abs(shapVal);
              const maxVal = Math.max(
                ...topContribs.map((c) => Math.abs(c.shap_value ?? c.shapValue ?? 0)),
                0.001
              );
              const barWidth = Math.min(Math.round((absVal / maxVal) * 100), 100);
              const dispVal =
                contrib.display_value ||
                contrib.displayValue ||
                (contrib.raw_value !== undefined ? String(contrib.raw_value) : "");

              return (
                <div
                  key={`${contrib.feature}-${idx}`}
                  className="rounded-lg border border-slate-100 bg-slate-50/70 p-3 text-xs"
                >
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <span className="font-semibold text-slate-900">{featName}</span>
                    <div className="flex items-center gap-1.5">
                      {dispVal && (
                        <span className="rounded bg-slate-200/80 px-2 py-0.5 text-[11px] font-medium text-slate-700">
                          {dispVal}
                        </span>
                      )}
                      <span
                        className={`rounded px-2 py-0.5 text-[11px] font-semibold ${
                          pushesPhish
                            ? "bg-red-100 text-red-700"
                            : "bg-teal-100 text-teal-700"
                        }`}
                      >
                        {pushesPhish ? "Towards Phishing" : "Towards Legitimate"} (
                        {shapVal > 0 ? "+" : ""}
                        {shapVal.toFixed(3)})
                      </span>
                    </div>
                  </div>
                  <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-slate-200">
                    <div
                      className={`h-full ${pushesPhish ? "bg-red-500" : "bg-teal-500"}`}
                      style={{ width: `${barWidth}%` }}
                    />
                  </div>
                  <p className="mt-2 text-[11px] leading-relaxed text-slate-600">
                    {contrib.explanation}
                  </p>
                </div>
              );
            })}
          </div>
        </div>
      )}

      <div className="grid gap-3 sm:grid-cols-3">
        <MiniStatus icon={Database} label="DNS" value={urlResult.dnsStatus ?? "Available"} />
        <MiniStatus icon={Globe2} label="Page" value={urlResult.htmlStatus ?? "Available"} />
        <MiniStatus icon={LockKeyhole} label="Host" value={urlResult.hostname ?? "N/A"} />
      </div>
    </div>
  );
}

interface CsvResultProps {
  csvChartData: Array<{ fill: string; name: string; value: number }>;
  csvLegitimatePercent: number;
  csvPhishingPercent: number;
  csvResult: CsvPrediction;
  previewColumns: string[];
}

function CsvResult({ csvChartData, csvLegitimatePercent, csvPhishingPercent, csvResult, previewColumns }: CsvResultProps) {
  return (
    <div className="space-y-5">
      <div className="flex flex-col justify-between gap-4 rounded-lg border border-slate-200 bg-slate-50 p-4 sm:flex-row sm:items-center">
        <div>
          <p className="text-sm text-slate-500">Batch prediction completed</p>
          <h3 className="break-all text-2xl font-semibold text-slate-950">{csvResult.fileName}</h3>
          <p className="mt-1 text-sm text-slate-500">{csvResult.outputMeta}</p>
        </div>
        <Button asChild variant="outline">
          <a href={apiUrl("/download-output")}>
            Download scored CSV
            <Download className="h-4 w-4" />
          </a>
        </Button>
      </div>
      <div className="rounded-lg border border-slate-200 bg-white p-4">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <p className="text-sm font-semibold text-slate-700">Scored file summary</p>
          <p className="text-sm text-slate-500">
            {csvPhishingPercent}% phishing / {csvLegitimatePercent}% legitimate
          </p>
        </div>
        <div className="flex h-3 overflow-hidden rounded-full bg-slate-100">
          <div className="bg-red-500" style={{ width: `${csvPhishingPercent}%` }} />
          <div className="bg-teal-500" style={{ width: `${csvLegitimatePercent}%` }} />
        </div>
      </div>
      <div className="grid gap-3 sm:grid-cols-4">
        <MiniStatus icon={FileSpreadsheet} label="Rows" value={csvResult.rows.toLocaleString()} />
        <MiniStatus icon={Grid3X3} label="Columns" value={csvResult.columns.toLocaleString()} />
        <MiniStatus icon={ShieldAlert} label="Phishing" value={csvResult.phishing.toLocaleString()} />
        <MiniStatus icon={ShieldCheck} label="Legitimate" value={csvResult.legitimate.toLocaleString()} />
      </div>
      <div className="grid gap-4 lg:grid-cols-[0.7fr_1.3fr]">
        <div className="rounded-lg border border-slate-200 bg-white p-4">
          <p className="mb-3 text-sm font-semibold text-slate-700">Prediction split</p>
          <ResponsiveContainer height={190} width="100%">
            <BarChart data={csvChartData}>
              <XAxis dataKey="name" stroke="#64748b" tickLine={false} />
              <YAxis allowDecimals={false} stroke="#64748b" tickLine={false} />
              <Tooltip />
              <Bar dataKey="value" radius={[6, 6, 0, 0]}>
                {csvChartData.map((entry) => (
                  <Cell fill={entry.fill} key={entry.name} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <CsvPreviewTable csvResult={csvResult} previewColumns={previewColumns} />
      </div>
    </div>
  );
}

function CsvPreviewTable({ csvResult, previewColumns }: { csvResult: CsvPrediction; previewColumns: string[] }) {
  return (
    <div className="overflow-hidden rounded-lg border border-slate-200 bg-white">
      <div className="border-b border-slate-200 px-4 py-3">
        <p className="text-sm font-semibold text-slate-700">Preview with predicted column</p>
        <p className="text-xs text-slate-500">Showing first {csvResult.preview.length} scored rows.</p>
      </div>
      <div className="max-h-64 overflow-auto">
        <table className="w-full min-w-[620px] text-left text-xs">
          <thead className="sticky top-0 bg-slate-50 text-slate-500">
            <tr>
              {previewColumns.map((column) => (
                <th className="border-b border-slate-200 px-3 py-2 font-semibold" key={column}>
                  {column}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {csvResult.preview.map((row, rowIndex) => (
              <tr className="hover:bg-slate-50" key={`${csvResult.fileName}-${rowIndex}`}>
                {previewColumns.map((column) => (
                  <td className="border-b border-slate-100 px-3 py-2 text-slate-700" key={column}>
                    {column === "predicted_column" ? (
                      <Badge
                        className={
                          Number(row[column]) === 1
                            ? "border-teal-200 bg-teal-50 text-teal-700"
                            : "border-red-200 bg-red-50 text-red-700"
                        }
                      >
                        {Number(row[column]) === 1 ? "Legitimate" : "Phishing"}
                      </Badge>
                    ) : (
                      String(row[column])
                    )}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function EmptyResult() {
  return (
    <div className="grid min-h-72 place-items-center rounded-lg border border-dashed border-slate-300 bg-slate-50 text-center">
      <div className="max-w-sm p-6">
        <LineChart className="mx-auto mb-4 h-10 w-10 text-teal-600" />
        <h3 className="text-xl font-semibold">Results will appear here</h3>
        <p className="mt-2 text-sm leading-6 text-slate-500">Run a URL or CSV prediction to see status, confidence, and explainability output.</p>
      </div>
    </div>
  );
}

export { ResultPanel };
