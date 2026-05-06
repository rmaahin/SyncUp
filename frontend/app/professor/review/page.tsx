"use client";

import { useCallback, useEffect, useState } from "react";
import {
  ApiError,
  generatePeerReviewForms,
  generateReports,
  getPeerReviewStatus,
  getTeamReport,
  type BiasFlag,
  type PeerReviewStatus,
  type StudentReport,
  type TeamReport,
} from "@/lib/api";
import { useSession } from "@/lib/session-context";

const SEVERITY_CLASSES: Record<string, string> = {
  high: "bg-red-100 border-red-400 text-red-900",
  medium: "bg-orange-100 border-orange-400 text-orange-900",
  low: "bg-yellow-100 border-yellow-400 text-yellow-900",
};

export default function ProfessorReviewPage() {
  const { projectId } = useSession();
  const [status, setStatus] = useState<PeerReviewStatus | null>(null);
  const [report, setReport] = useState<TeamReport | null>(null);
  const [generatingForms, setGeneratingForms] = useState(false);
  const [generatingReport, setGeneratingReport] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadStatus = useCallback(async () => {
    try {
      const s = await getPeerReviewStatus(projectId);
      setStatus(s);
    } catch (err) {
      if (err instanceof ApiError) setError(err.detail);
    }
  }, [projectId]);

  const loadReport = useCallback(async () => {
    try {
      const r = await getTeamReport(projectId);
      setReport(r);
    } catch (err) {
      if (err instanceof ApiError && err.status === 404) {
        setReport(null);
      } else if (err instanceof ApiError) {
        setError(err.detail);
      }
    }
  }, [projectId]);

  useEffect(() => {
    void loadStatus();
    void loadReport();
  }, [loadStatus, loadReport]);

  const handleGenerateForms = async () => {
    setGeneratingForms(true);
    setError(null);
    try {
      await generatePeerReviewForms(projectId);
      await loadStatus();
    } catch (err) {
      if (err instanceof ApiError) setError(err.detail);
      else setError("Failed to generate forms");
    } finally {
      setGeneratingForms(false);
    }
  };

  const handleGenerateReport = async () => {
    setGeneratingReport(true);
    setError(null);
    try {
      const r = await generateReports(projectId);
      setReport(r);
    } catch (err) {
      if (err instanceof ApiError) setError(err.detail);
      else setError("Failed to generate report");
    } finally {
      setGeneratingReport(false);
    }
  };

  return (
    <div className="mx-auto max-w-5xl space-y-6 py-6">
      <header>
        <h1 className="text-2xl font-semibold text-slate-900">
          Peer Review &amp; Final Report
        </h1>
        <p className="mt-1 text-sm text-muted">Project {projectId}</p>
      </header>

      {error && (
        <div className="rounded border border-red-300 bg-red-50 p-3 text-sm text-red-800">
          {error}
        </div>
      )}

      <section className="rounded-lg border border-border bg-surface p-5">
        <h2 className="text-lg font-semibold text-slate-900">Submission status</h2>
        {status ? (
          <div className="mt-2 flex flex-wrap gap-6 text-sm">
            <div>
              <span className="font-medium">Submitted:</span>{" "}
              {status.submitted.length} / {status.total}
            </div>
            <div>
              <span className="font-medium">Pending:</span>{" "}
              {status.pending.length === 0 ? "—" : status.pending.join(", ")}
            </div>
          </div>
        ) : (
          <p className="text-sm text-muted">Loading…</p>
        )}
      </section>

      <section className="flex flex-wrap gap-3">
        <button
          type="button"
          onClick={handleGenerateForms}
          disabled={generatingForms}
          className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
        >
          {generatingForms ? "Generating…" : "Generate peer-review forms"}
        </button>
        <button
          type="button"
          onClick={handleGenerateReport}
          disabled={generatingReport}
          className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
          title="May take up to a minute (LLM call per student)"
        >
          {generatingReport ? "Generating final report…" : "Generate final report"}
        </button>
      </section>

      {report && (
        <>
          <BiasBanners flags={report.bias_flags} />
          <TeamSummary report={report} />
          <StudentList reports={report.student_reports} />
        </>
      )}
    </div>
  );
}

function BiasBanners({ flags }: { flags: BiasFlag[] }) {
  if (flags.length === 0) return null;
  return (
    <section className="space-y-2">
      <h2 className="text-lg font-semibold text-slate-900">Bias flags</h2>
      {flags.map((f, i) => (
        <div
          key={i}
          className={`rounded border-l-4 p-3 text-sm ${SEVERITY_CLASSES[f.severity] ?? SEVERITY_CLASSES.low}`}
        >
          <div className="font-medium">
            [{f.severity.toUpperCase()}] {f.flag_type}
          </div>
          <div className="mt-1">{f.description}</div>
          <div className="mt-1 text-xs">
            reviewer: <code>{f.reviewer_id}</code>
            {f.reviewee_id && (
              <>
                {" · "}reviewee: <code>{f.reviewee_id}</code>
              </>
            )}
          </div>
        </div>
      ))}
    </section>
  );
}

function TeamSummary({ report }: { report: TeamReport }) {
  return (
    <section className="rounded-lg border border-border bg-surface p-5">
      <h2 className="text-lg font-semibold text-slate-900">Team summary</h2>
      {report.team_narrative && (
        <p className="mt-2 text-sm text-slate-800">{report.team_narrative}</p>
      )}
      <dl className="mt-4 grid grid-cols-2 gap-3 text-sm md:grid-cols-3">
        {Object.entries(report.team_metrics).map(([k, v]) => (
          <div key={k}>
            <dt className="text-xs text-muted">{k}</dt>
            <dd className="font-mono">{formatVal(v)}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

function StudentList({ reports }: { reports: StudentReport[] }) {
  return (
    <section className="space-y-3">
      <h2 className="text-lg font-semibold text-slate-900">Per-student reports</h2>
      {reports.map((r) => (
        <details
          key={r.student_id}
          className="rounded-lg border border-border bg-surface p-4"
        >
          <summary className="cursor-pointer font-medium text-slate-900">
            {r.student_id}
          </summary>
          <div className="mt-3 space-y-3 text-sm">
            {r.narrative && <p>{r.narrative}</p>}
            {r.strengths.length > 0 && (
              <div>
                <h4 className="font-medium">Strengths</h4>
                <ul className="ml-5 list-disc">
                  {r.strengths.map((s, i) => (
                    <li key={i}>{s}</li>
                  ))}
                </ul>
              </div>
            )}
            {r.areas_for_improvement.length > 0 && (
              <div>
                <h4 className="font-medium">Areas for improvement</h4>
                <ul className="ml-5 list-disc">
                  {r.areas_for_improvement.map((s, i) => (
                    <li key={i}>{s}</li>
                  ))}
                </ul>
              </div>
            )}
            <div>
              <h4 className="font-medium">Metrics</h4>
              <dl className="mt-1 grid grid-cols-2 gap-2 md:grid-cols-3">
                {Object.entries(r.metrics).map(([k, v]) => (
                  <div key={k}>
                    <dt className="text-xs text-muted">{k}</dt>
                    <dd className="font-mono">{formatVal(v)}</dd>
                  </div>
                ))}
              </dl>
            </div>
            {r.peer_summary && r.peer_summary.review_count > 0 && (
              <div>
                <h4 className="font-medium">
                  Peer ratings (n={r.peer_summary.review_count})
                </h4>
                <dl className="mt-1 grid grid-cols-2 gap-2 md:grid-cols-3">
                  <div>
                    <dt className="text-xs text-muted">overall_avg</dt>
                    <dd className="font-mono">
                      {r.peer_summary.overall_avg.toFixed(2)}
                    </dd>
                  </div>
                  {Object.entries(r.peer_summary.avg_per_dimension).map(([k, v]) => (
                    <div key={k}>
                      <dt className="text-xs text-muted">{k}</dt>
                      <dd className="font-mono">{v.toFixed(2)}</dd>
                    </div>
                  ))}
                </dl>
              </div>
            )}
          </div>
        </details>
      ))}
    </section>
  );
}

function formatVal(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(2);
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}
