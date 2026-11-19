import { CheckCircle2 } from "lucide-react";

import { Badge } from "../ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import { formatConfidence, formatScanTime } from "../../lib/format";
import type { UrlPrediction } from "../../lib/types";

interface HistoryListProps {
  history: UrlPrediction[];
  latestUrl?: string;
  variant?: "compact" | "full";
}

function HistoryList({ history, latestUrl, variant = "compact" }: HistoryListProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{variant === "full" ? "Saved URL prediction history" : "Recent URL checks"}</CardTitle>
      </CardHeader>
      <CardContent>
        {history.length ? (
          <div className="space-y-3">
            {history.map((item, index) => (
              <HistoryRow item={item} key={item.historyId ?? `${item.url}-${index}`} variant={variant} />
            ))}
          </div>
        ) : (
          <div className="grid min-h-64 place-items-center rounded-lg border border-dashed border-slate-300 bg-slate-50 text-center">
            <div className="max-w-sm p-6">
              <CheckCircle2 className="mx-auto mb-4 h-10 w-10 text-slate-400" />
              <h3 className="font-semibold">No predictions yet</h3>
              <p className="mt-2 text-sm text-slate-500">Analyze a URL to populate this history.</p>
            </div>
          </div>
        )}
        {latestUrl ? <p className="mt-4 text-xs text-slate-500">Latest checked URL: {latestUrl}</p> : null}
      </CardContent>
    </Card>
  );
}

function HistoryRow({ item, variant }: { item: UrlPrediction; variant: "compact" | "full" }) {
  const isLegitimate =
    item.prediction === "Legitimate" ||
    item.label === "Legitimate" ||
    item.label === 1 ||
    item.prediction === 1;
  const displayLabel =
    typeof item.prediction === "string"
      ? item.prediction
      : (typeof item.label === "string"
          ? item.label
          : (item.label === 1 || item.prediction === 1 ? "Legitimate" : "Phishing"));
  const confidencePercent = formatConfidence(item.confidence);
  const suspiciousSignals = item.signals?.suspicious ?? 0;

  return (
    <div className="flex flex-col justify-between gap-3 rounded-lg border border-slate-200 p-4 sm:flex-row sm:items-center">
      <div className="min-w-0">
        <div className="flex flex-wrap items-center gap-2">
          <p className="font-medium">{item.hostname || "URL"}</p>
          <span className="text-xs text-slate-400">{formatScanTime(item.createdAt)}</span>
        </div>
        <p className="max-w-md truncate text-sm text-slate-500">{item.url}</p>
        <p className="mt-1 text-xs text-slate-500">
          {suspiciousSignals} suspicious signals / {confidencePercent}% confidence
        </p>
        {variant === "full" && item.suspiciousFeatures?.length ? (
          <p className="mt-2 text-xs text-red-600">Suspicious: {item.suspiciousFeatures.slice(0, 5).join(", ")}</p>
        ) : null}
      </div>
      <Badge className={isLegitimate ? "border-teal-200 bg-teal-50 text-teal-700" : "border-red-200 bg-red-50 text-red-700"}>
        {displayLabel}
      </Badge>
    </div>
  );
}

export { HistoryList };
