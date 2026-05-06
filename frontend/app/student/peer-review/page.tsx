"use client";

import { useEffect, useState } from "react";
import { PeerReviewFormCard } from "@/components/PeerReviewForm";
import {
  ApiError,
  getPeerReviewForm,
  submitPeerReview,
  type PeerReviewForm,
} from "@/lib/api";
import { useSession } from "@/lib/session-context";

type PerTeammateValue = {
  ratings: Record<string, number>;
  comments: Record<string, string>;
};

export default function StudentPeerReviewPage() {
  const { projectId, userId } = useSession();
  const [form, setForm] = useState<PeerReviewForm | null>(null);
  const [values, setValues] = useState<Record<string, PerTeammateValue>>({});
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [submitted, setSubmitted] = useState(false);
  const [alreadySubmitted, setAlreadySubmitted] = useState(false);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError(null);
    getPeerReviewForm(projectId, userId)
      .then((f) => {
        if (!active) return;
        setForm(f);
        const init: Record<string, PerTeammateValue> = {};
        for (const t of f.teammates) {
          init[t.id] = { ratings: {}, comments: {} };
        }
        setValues(init);
      })
      .catch((err) => {
        if (!active) return;
        if (err instanceof ApiError && err.status === 409) {
          setAlreadySubmitted(true);
        } else if (err instanceof ApiError) {
          setError(err.detail);
        } else {
          setError("Could not load form");
        }
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [projectId, userId]);

  const allRated =
    form !== null &&
    form.teammates.every((t) =>
      form.dimensions.every((d) => typeof values[t.id]?.ratings[d.key] === "number"),
    );

  const onSubmit = async () => {
    if (!form) return;
    setSubmitting(true);
    setError(null);
    try {
      await submitPeerReview(projectId, {
        reviewer_id: userId,
        reviews: form.teammates.map((t) => ({
          reviewee_id: t.id,
          ratings: values[t.id].ratings,
          comments: values[t.id].comments,
        })),
      });
      setSubmitted(true);
    } catch (err) {
      if (err instanceof ApiError && err.status === 409) {
        setAlreadySubmitted(true);
      } else if (err instanceof ApiError) {
        setError(err.detail);
      } else {
        setError("Submission failed");
      }
    } finally {
      setSubmitting(false);
    }
  };

  if (loading) {
    return <p className="py-8 text-center text-muted">Loading peer review form…</p>;
  }
  if (alreadySubmitted) {
    return (
      <div className="mx-auto max-w-xl py-12 text-center">
        <h1 className="text-2xl font-semibold text-slate-900">Already submitted</h1>
        <p className="mt-2 text-sm text-muted">
          You have already submitted your peer review. Thanks!
        </p>
      </div>
    );
  }
  if (submitted) {
    return (
      <div className="mx-auto max-w-xl py-12 text-center">
        <h1 className="text-2xl font-semibold text-slate-900">Thank you</h1>
        <p className="mt-2 text-sm text-muted">
          Your peer review has been recorded.
        </p>
      </div>
    );
  }
  if (error && !form) {
    return <p className="py-8 text-center text-red-600">{error}</p>;
  }
  if (!form) return null;

  const remaining = form.teammates.filter((t) =>
    form.dimensions.some((d) => typeof values[t.id]?.ratings[d.key] !== "number"),
  ).length;

  return (
    <div className="mx-auto max-w-3xl space-y-6 py-8">
      <header>
        <h1 className="text-2xl font-semibold text-slate-900">Peer Review</h1>
        <p className="mt-1 text-sm text-muted">
          Rate each teammate on the five dimensions. Comments are optional.
        </p>
      </header>

      {form.teammates.map((t) => (
        <PeerReviewFormCard
          key={t.id}
          teammate={t}
          dimensions={form.dimensions}
          value={values[t.id] ?? { ratings: {}, comments: {} }}
          onChange={(next) => setValues((prev) => ({ ...prev, [t.id]: next }))}
        />
      ))}

      {error && <p className="text-sm text-red-600">{error}</p>}

      <div className="sticky bottom-0 flex items-center justify-between rounded-lg border border-border bg-surface p-4 shadow">
        <span className="text-sm text-muted">
          {allRated ? "All teammates rated" : `${remaining} teammate(s) remaining`}
        </span>
        <button
          type="button"
          disabled={!allRated || submitting}
          onClick={onSubmit}
          className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
        >
          {submitting ? "Submitting…" : "Submit review"}
        </button>
      </div>
    </div>
  );
}
