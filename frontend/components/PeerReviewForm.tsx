"use client";

import type { PeerReviewDimension, PeerReviewTeammate } from "@/lib/api";

interface FormValue {
  ratings: Record<string, number>;
  comments: Record<string, string>;
}

interface Props {
  teammate: PeerReviewTeammate;
  dimensions: PeerReviewDimension[];
  value: FormValue;
  onChange: (next: FormValue) => void;
}

export function PeerReviewFormCard({ teammate, dimensions, value, onChange }: Props) {
  const setRating = (dim: string, n: number) => {
    onChange({ ...value, ratings: { ...value.ratings, [dim]: n } });
  };
  const setComment = (dim: string, text: string) => {
    onChange({ ...value, comments: { ...value.comments, [dim]: text } });
  };

  return (
    <section className="rounded-lg border border-border bg-surface p-5 shadow-sm">
      <header className="mb-4">
        <h3 className="text-lg font-semibold text-slate-900">{teammate.name}</h3>
        {teammate.assigned_tasks.length > 0 && (
          <p className="mt-1 text-xs text-muted">
            Assigned tasks: {teammate.assigned_tasks.join(", ")}
          </p>
        )}
      </header>
      <div className="space-y-4">
        {dimensions.map((d) => {
          const current = value.ratings[d.key];
          return (
            <div key={d.key} className="space-y-2">
              <label className="block text-sm font-medium text-slate-800">
                {d.question}
              </label>
              {d.description && (
                <p className="text-xs text-muted">{d.description}</p>
              )}
              <div className="flex items-center gap-3">
                <input
                  type="range"
                  min={1}
                  max={5}
                  step={1}
                  value={current ?? 3}
                  onChange={(e) => setRating(d.key, Number(e.target.value))}
                  className="flex-1"
                />
                <span className="w-8 text-center text-sm font-mono">
                  {current ?? "—"}
                </span>
              </div>
              <textarea
                rows={2}
                placeholder="Optional comment"
                value={value.comments[d.key] ?? ""}
                onChange={(e) => setComment(d.key, e.target.value)}
                className="w-full rounded border border-border bg-bg px-2 py-1 text-sm"
              />
            </div>
          );
        })}
      </div>
    </section>
  );
}

export default PeerReviewFormCard;
