// Typed fetch helpers for the SyncUp backend.

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.status = status;
    this.detail = detail;
    this.name = "ApiError";
  }
}

export async function apiFetch<T>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init.headers ?? {}),
    },
  });
  if (!res.ok) {
    let detail: string = res.statusText;
    try {
      const body = (await res.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
      else if (body.detail !== undefined) detail = JSON.stringify(body.detail);
    } catch {
      // ignore JSON parse errors
    }
    throw new ApiError(res.status, detail);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

// ---------------------------------------------------------------------------
// Onboarding
// ---------------------------------------------------------------------------

export type OAuthProvider = "github" | "google" | "trello";

export interface BlackoutPeriod {
  start: string;
  end: string;
}

export interface CreateProfileRequest {
  name: string;
  email: string;
  timezone: string;
  skills: Record<string, number>;
  availability_hours_per_week: number;
  preferred_times: string[];
  blackout_periods: BlackoutPeriod[];
  github_username: string;
  google_email: string;
  trello_username: string;
}

export const createProfile = (body: CreateProfileRequest) =>
  apiFetch<{ student_id: string }>("/api/onboarding/profile", {
    method: "POST",
    body: JSON.stringify(body),
  });

export const oauthConnect = (provider: OAuthProvider) =>
  apiFetch<{ authorize_url: string }>(`/api/onboarding/oauth/${provider}/start`, {
    method: "POST",
  });

// ---------------------------------------------------------------------------
// Peer review + reports (Phase 12)
// ---------------------------------------------------------------------------

export interface PeerReviewDimension {
  key: string;
  question: string;
  description: string;
}

export interface PeerReviewTeammate {
  id: string;
  name: string;
  assigned_tasks: string[];
}

export interface PeerReviewTemplate {
  dimensions: PeerReviewDimension[];
  forms_by_student: Record<string, { teammates: PeerReviewTeammate[] }>;
}

export interface PeerReviewForm {
  dimensions: PeerReviewDimension[];
  teammates: PeerReviewTeammate[];
}

export interface PeerReviewSubmission {
  reviewer_id: string;
  reviews: {
    reviewee_id: string;
    ratings: Record<string, number>;
    comments: Record<string, string>;
  }[];
}

export interface PeerReviewStatus {
  submitted: string[];
  pending: string[];
  total: number;
}

export interface BiasFlag {
  flag_type: "outlier_reviewer" | "targeted_low" | "inflation" | "retaliation" | string;
  reviewer_id: string;
  reviewee_id: string | null;
  description: string;
  severity: "low" | "medium" | "high" | string;
}

export interface PeerReviewSummary {
  student_id: string;
  avg_per_dimension: Record<string, number>;
  overall_avg: number;
  std_dev_per_dimension: Record<string, number>;
  review_count: number;
}

export interface StudentReport {
  student_id: string;
  metrics: Record<string, unknown>;
  peer_summary: PeerReviewSummary | null;
  peer_bias_flags: BiasFlag[];
  narrative: string;
  strengths: string[];
  areas_for_improvement: string[];
}

export interface TeamReport {
  project_id: string;
  completion_pct: number;
  team_metrics: Record<string, unknown>;
  student_reports: StudentReport[];
  bias_flags: BiasFlag[];
  team_narrative: string;
  generated_at: string;
}

export const generatePeerReviewForms = (pid: string) =>
  apiFetch<PeerReviewTemplate>(`/api/peer-review/${pid}/generate`, { method: "POST" });

export const getPeerReviewForm = (pid: string, sid: string) =>
  apiFetch<PeerReviewForm>(`/api/peer-review/${pid}/form/${sid}`);

export const submitPeerReview = (pid: string, payload: PeerReviewSubmission) =>
  apiFetch<{ count: number }>(`/api/peer-review/${pid}/submit`, {
    method: "POST",
    body: JSON.stringify(payload),
  });

export const getPeerReviewStatus = (pid: string) =>
  apiFetch<PeerReviewStatus>(`/api/peer-review/${pid}/status`);

export const generateReports = (pid: string) =>
  apiFetch<TeamReport>(`/api/reports/${pid}/generate`, { method: "POST" });

export const getTeamReport = (pid: string) =>
  apiFetch<TeamReport>(`/api/reports/${pid}/team`);

export const getBiasFlags = (pid: string) =>
  apiFetch<{ bias_flags: BiasFlag[] }>(`/api/reports/${pid}/bias-flags`);
