"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import { useRouter } from "next/navigation";
import { useMemo, useState } from "react";
import {
  Controller,
  FormProvider,
  useFieldArray,
  useForm,
  useFormContext,
  type FieldPath,
} from "react-hook-form";
import { z } from "zod";
import {
  ApiError,
  createProfile,
  oauthConnect,
  type CreateProfileRequest,
  type OAuthProvider,
} from "@/lib/api";
import { useSession } from "@/lib/session-context";

// ---------------------------------------------------------------------------
// Schema
// ---------------------------------------------------------------------------

const PREDEFINED_SKILLS = [
  "Frontend",
  "Backend",
  "Data Analysis",
  "ML/AI",
  "Technical Writing",
  "Design",
  "Database",
  "DevOps",
  "Research",
  "Testing",
] as const;

const PREFERRED_TIMES = ["Morning", "Afternoon", "Evening", "Flexible"] as const;

const blackoutSchema = z
  .object({ start: z.string().min(1, "Start required"), end: z.string().min(1, "End required") })
  .refine((b) => b.end >= b.start, {
    message: "End must be on or after start",
    path: ["end"],
  });

const schema = z.object({
  basic: z.object({
    name: z.string().min(1, "Name is required"),
    email: z.string().email("Enter a valid email"),
    timezone: z.string().min(1, "Timezone is required"),
  }),
  skills: z
    .array(
      z.object({
        name: z.string().min(1, "Skill name required"),
        proficiency: z.number().min(0).max(1),
      }),
    )
    .min(1, "Add at least one skill"),
  availability: z.object({
    hoursPerWeek: z.coerce.number().int().min(1, "At least 1").max(40, "At most 40"),
    preferredTimes: z.array(z.string()).min(1, "Pick at least one preferred time"),
    blackouts: z.array(blackoutSchema),
  }),
  connections: z.object({
    google: z.boolean(),
    github: z.boolean(),
    trello: z.boolean(),
  }),
});

type FormValues = z.infer<typeof schema>;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function getTimezoneList(): string[] {
  const intlAny = Intl as unknown as {
    supportedValuesOf?: (k: string) => string[];
  };
  if (typeof intlAny.supportedValuesOf === "function") {
    try {
      return intlAny.supportedValuesOf("timeZone");
    } catch {
      // fallthrough
    }
  }
  return ["UTC", "America/New_York", "America/Los_Angeles", "Europe/London", "Asia/Tokyo"];
}

function detectTimezone(): string {
  try {
    return Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC";
  } catch {
    return "UTC";
  }
}

function proficiencyLabel(v: number): string {
  if (v < 0.34) return "Beginner";
  if (v < 0.67) return "Intermediate";
  return "Advanced";
}

const STEP_FIELDS: Record<number, FieldPath<FormValues>[]> = {
  1: ["basic.name", "basic.email", "basic.timezone"],
  2: ["skills"],
  3: [
    "availability.hoursPerWeek",
    "availability.preferredTimes",
    "availability.blackouts",
  ],
  4: [],
  5: [],
};

const TOTAL_STEPS = 5;

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function OnboardingPage() {
  const router = useRouter();
  const { userId } = useSession();
  const [step, setStep] = useState(1);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const methods = useForm<FormValues>({
    resolver: zodResolver(schema),
    mode: "onTouched",
    defaultValues: {
      basic: { name: "", email: "", timezone: detectTimezone() },
      skills: [],
      availability: {
        hoursPerWeek: 10,
        preferredTimes: [],
        blackouts: [],
      },
      connections: { google: false, github: false, trello: false },
    },
  });

  const goNext = async () => {
    const fields = STEP_FIELDS[step] ?? [];
    const ok = fields.length === 0 ? true : await methods.trigger(fields);
    if (ok) setStep((s) => Math.min(TOTAL_STEPS, s + 1));
  };

  const goBack = () => setStep((s) => Math.max(1, s - 1));

  const onSubmit = async (values: FormValues) => {
    setSubmitting(true);
    setSubmitError(null);
    try {
      const body: CreateProfileRequest = {
        name: values.basic.name,
        email: values.basic.email,
        timezone: values.basic.timezone,
        skills: Object.fromEntries(
          values.skills.map((s) => [s.name, Number(s.proficiency.toFixed(2))]),
        ),
        availability_hours_per_week: values.availability.hoursPerWeek,
        preferred_times: values.availability.preferredTimes,
        blackout_periods: values.availability.blackouts.map((b) => ({
          start: new Date(b.start).toISOString(),
          end: new Date(b.end).toISOString(),
        })),
        github_username: "",
        google_email: "",
        trello_username: "",
      };
      await createProfile(body);
      router.push("/student/dashboard");
    } catch (err) {
      const msg =
        err instanceof ApiError
          ? err.message
          : err instanceof Error
          ? err.message
          : "Submission failed";
      setSubmitError(msg);
      setSubmitting(false);
    }
  };

  return (
    <div className="mx-auto max-w-2xl">
      <header className="mb-6">
        <p className="text-sm text-muted">
          Welcome — let&apos;s set up your profile so SyncUp can match you to
          the right tasks.
        </p>
        <p className="mt-1 text-xs text-muted">
          Demo user id: <span className="font-mono">{userId}</span>
        </p>
      </header>

      <StepIndicator current={step} total={TOTAL_STEPS} />

      <FormProvider {...methods}>
        <form
          onSubmit={methods.handleSubmit(onSubmit)}
          className="mt-6 rounded-xl border border-border bg-surface p-8 shadow-sm"
        >
          {step === 1 && <StepBasicInfo />}
          {step === 2 && <StepSkills />}
          {step === 3 && <StepAvailability />}
          {step === 4 && <StepConnections />}
          {step === 5 && <StepReview />}

          {submitError && (
            <div className="mt-6 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
              {submitError}
            </div>
          )}

          <div className="mt-8 flex items-center justify-between border-t border-border pt-6">
            <button
              type="button"
              onClick={goBack}
              disabled={step === 1 || submitting}
              className="rounded-lg px-4 py-2 text-sm font-medium text-slate-700 hover:bg-bg disabled:opacity-40"
            >
              ← Back
            </button>
            {step < TOTAL_STEPS ? (
              <button
                type="button"
                onClick={goNext}
                className="rounded-lg bg-accent px-5 py-2 text-sm font-medium text-white shadow-sm hover:bg-accent-hover"
              >
                Next →
              </button>
            ) : (
              <button
                type="submit"
                disabled={submitting}
                className="rounded-lg bg-accent px-5 py-2 text-sm font-medium text-white shadow-sm hover:bg-accent-hover disabled:opacity-60"
              >
                {submitting ? "Submitting…" : "Submit"}
              </button>
            )}
          </div>
        </form>
      </FormProvider>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Step indicator
// ---------------------------------------------------------------------------

const STEP_LABELS = [
  "Basics",
  "Skills",
  "Availability",
  "Connect",
  "Review",
];

function StepIndicator({ current, total }: { current: number; total: number }) {
  return (
    <div>
      <div className="mb-2 flex items-baseline justify-between">
        <span className="text-sm font-medium text-slate-900">
          Step {current} of {total}: {STEP_LABELS[current - 1]}
        </span>
        <span className="text-xs text-muted">
          {Math.round(((current - 1) / (total - 1)) * 100)}%
        </span>
      </div>
      <div className="flex gap-2">
        {Array.from({ length: total }).map((_, i) => (
          <div
            key={i}
            className={`h-1.5 flex-1 rounded-full ${
              i < current ? "bg-accent" : "bg-border"
            }`}
          />
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Step 1 — Basics
// ---------------------------------------------------------------------------

function StepBasicInfo() {
  const { register, formState: { errors } } = useFormContext<FormValues>();
  const timezones = useMemo(() => getTimezoneList(), []);
  return (
    <div className="space-y-5">
      <h2 className="text-lg font-semibold text-slate-900">Tell us about you</h2>

      <Field label="Full name" error={errors.basic?.name?.message}>
        <input
          {...register("basic.name")}
          className="w-full rounded-lg border border-border px-3 py-2 text-sm focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/20"
          placeholder="Ada Lovelace"
        />
      </Field>

      <Field label="Email" error={errors.basic?.email?.message}>
        <input
          type="email"
          {...register("basic.email")}
          className="w-full rounded-lg border border-border px-3 py-2 text-sm focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/20"
          placeholder="ada@university.edu"
        />
      </Field>

      <Field label="Timezone" error={errors.basic?.timezone?.message}>
        <select
          {...register("basic.timezone")}
          className="w-full rounded-lg border border-border bg-surface px-3 py-2 text-sm focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/20"
        >
          {timezones.map((tz) => (
            <option key={tz} value={tz}>
              {tz}
            </option>
          ))}
        </select>
      </Field>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Step 2 — Skills
// ---------------------------------------------------------------------------

function StepSkills() {
  const { control, formState: { errors }, watch, register } =
    useFormContext<FormValues>();
  const { fields, append, remove } = useFieldArray({ control, name: "skills" });
  const [customName, setCustomName] = useState("");

  const skills = watch("skills");
  const selectedNames = new Set(skills.map((s) => s.name));

  const togglePredefined = (name: string) => {
    if (selectedNames.has(name)) {
      const idx = skills.findIndex((s) => s.name === name);
      if (idx >= 0) remove(idx);
    } else {
      append({ name, proficiency: 0.5 });
    }
  };

  const addCustom = () => {
    const trimmed = customName.trim();
    if (!trimmed || selectedNames.has(trimmed)) return;
    append({ name: trimmed, proficiency: 0.5 });
    setCustomName("");
  };

  return (
    <div className="space-y-5">
      <h2 className="text-lg font-semibold text-slate-900">What can you do?</h2>
      <p className="text-sm text-muted">
        Pick the skills you can contribute. Slide to set proficiency.
      </p>

      <div className="flex flex-wrap gap-2">
        {PREDEFINED_SKILLS.map((s) => {
          const active = selectedNames.has(s);
          return (
            <button
              key={s}
              type="button"
              onClick={() => togglePredefined(s)}
              className={`rounded-full border px-3 py-1.5 text-sm transition ${
                active
                  ? "border-accent bg-accent text-white"
                  : "border-border bg-surface text-slate-700 hover:border-accent"
              }`}
            >
              {s}
            </button>
          );
        })}
      </div>

      <div className="flex gap-2">
        <input
          value={customName}
          onChange={(e) => setCustomName(e.target.value)}
          placeholder="Add a custom skill…"
          className="flex-1 rounded-lg border border-border px-3 py-2 text-sm focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/20"
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              addCustom();
            }
          }}
        />
        <button
          type="button"
          onClick={addCustom}
          className="rounded-lg border border-border bg-surface px-3 py-2 text-sm font-medium hover:bg-bg"
        >
          Add
        </button>
      </div>

      {fields.length > 0 && (
        <div className="space-y-3 rounded-lg border border-border bg-bg p-4">
          {fields.map((field, idx) => {
            const value = skills[idx]?.proficiency ?? 0.5;
            return (
              <div key={field.id} className="flex items-center gap-4">
                <div className="w-32 truncate text-sm font-medium text-slate-900">
                  {field.name}
                </div>
                <Controller
                  control={control}
                  name={`skills.${idx}.proficiency`}
                  render={({ field: f }) => (
                    <input
                      type="range"
                      min={0}
                      max={1}
                      step={0.1}
                      value={f.value}
                      onChange={(e) => f.onChange(parseFloat(e.target.value))}
                      className="flex-1 accent-accent"
                    />
                  )}
                />
                <div className="w-24 text-right text-xs text-muted">
                  {proficiencyLabel(value)}
                </div>
                <button
                  type="button"
                  onClick={() => remove(idx)}
                  className="text-xs text-muted hover:text-red-600"
                  aria-label={`Remove ${field.name}`}
                >
                  ✕
                </button>
                <input type="hidden" {...register(`skills.${idx}.name`)} />
              </div>
            );
          })}
        </div>
      )}

      {errors.skills?.message && (
        <p className="text-sm text-red-600">{errors.skills.message}</p>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Step 3 — Availability
// ---------------------------------------------------------------------------

function StepAvailability() {
  const { register, control, watch, setValue, formState: { errors } } =
    useFormContext<FormValues>();
  const { fields, append, remove } = useFieldArray({
    control,
    name: "availability.blackouts",
  });
  const preferred = watch("availability.preferredTimes");

  const togglePreferred = (t: string) => {
    const next = preferred.includes(t)
      ? preferred.filter((x) => x !== t)
      : [...preferred, t];
    setValue("availability.preferredTimes", next, { shouldValidate: true });
  };

  return (
    <div className="space-y-5">
      <h2 className="text-lg font-semibold text-slate-900">When can you work?</h2>

      <Field
        label="Hours per week"
        error={errors.availability?.hoursPerWeek?.message}
      >
        <input
          type="number"
          min={1}
          max={40}
          {...register("availability.hoursPerWeek", { valueAsNumber: true })}
          className="w-32 rounded-lg border border-border px-3 py-2 text-sm focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/20"
        />
      </Field>

      <Field
        label="Preferred working times"
        error={errors.availability?.preferredTimes?.message}
      >
        <div className="flex flex-wrap gap-2">
          {PREFERRED_TIMES.map((t) => {
            const active = preferred.includes(t);
            return (
              <button
                key={t}
                type="button"
                onClick={() => togglePreferred(t)}
                className={`rounded-full border px-3 py-1.5 text-sm transition ${
                  active
                    ? "border-accent bg-accent text-white"
                    : "border-border bg-surface text-slate-700 hover:border-accent"
                }`}
              >
                {t}
              </button>
            );
          })}
        </div>
      </Field>

      <div>
        <div className="mb-2 flex items-center justify-between">
          <label className="block text-sm font-medium text-slate-900">
            Blackout periods
          </label>
          <button
            type="button"
            onClick={() => append({ start: "", end: "" })}
            className="text-sm font-medium text-accent hover:text-accent-hover"
          >
            + Add blackout
          </button>
        </div>
        <p className="mb-3 text-xs text-muted">
          Exam weeks, travel, or anything else that should keep tasks off your
          plate.
        </p>

        {fields.length === 0 && (
          <div className="rounded-lg border border-dashed border-border bg-bg px-4 py-6 text-center text-sm text-muted">
            No blackout periods yet.
          </div>
        )}

        <div className="space-y-3">
          {fields.map((field, idx) => {
            const blackoutErrors = errors.availability?.blackouts?.[idx];
            return (
              <div
                key={field.id}
                className="rounded-lg border border-border bg-surface p-3"
              >
                <div className="flex flex-wrap items-end gap-3">
                  <div>
                    <label className="block text-xs font-medium text-muted">
                      Start
                    </label>
                    <input
                      type="date"
                      {...register(`availability.blackouts.${idx}.start`)}
                      className="rounded-lg border border-border px-3 py-1.5 text-sm focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/20"
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-medium text-muted">
                      End
                    </label>
                    <input
                      type="date"
                      {...register(`availability.blackouts.${idx}.end`)}
                      className="rounded-lg border border-border px-3 py-1.5 text-sm focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/20"
                    />
                  </div>
                  <button
                    type="button"
                    onClick={() => remove(idx)}
                    className="ml-auto text-sm text-muted hover:text-red-600"
                  >
                    Remove
                  </button>
                </div>
                {(blackoutErrors?.start?.message ||
                  blackoutErrors?.end?.message ||
                  blackoutErrors?.root?.message) && (
                  <p className="mt-2 text-xs text-red-600">
                    {blackoutErrors?.start?.message ||
                      blackoutErrors?.end?.message ||
                      blackoutErrors?.root?.message}
                  </p>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Step 4 — Connections
// ---------------------------------------------------------------------------

function StepConnections() {
  const { watch, setValue } = useFormContext<FormValues>();
  const connections = watch("connections");
  const [pending, setPending] = useState<OAuthProvider | null>(null);
  const [error, setError] = useState<string | null>(null);

  const onConnect = async (provider: OAuthProvider) => {
    setPending(provider);
    setError(null);
    try {
      await oauthConnect(provider);
      setValue(`connections.${provider}`, true, { shouldDirty: true });
    } catch (err) {
      setError(err instanceof Error ? err.message : "OAuth failed");
    } finally {
      setPending(null);
    }
  };

  const providers: { id: OAuthProvider; label: string; description: string }[] = [
    { id: "google", label: "Google", description: "Calendar + Docs" },
    { id: "github", label: "GitHub", description: "Commits & pull requests" },
    { id: "trello", label: "Trello", description: "Task board" },
  ];

  return (
    <div className="space-y-5">
      <h2 className="text-lg font-semibold text-slate-900">Connect your accounts</h2>
      <p className="text-sm text-muted">
        SyncUp uses these to track contributions and schedule meetings. Optional
        for the demo — you can skip and connect later.
      </p>

      <div className="space-y-3">
        {providers.map((p) => {
          const connected = connections[p.id];
          return (
            <div
              key={p.id}
              className="flex items-center justify-between rounded-lg border border-border bg-surface px-4 py-3"
            >
              <div>
                <div className="font-medium text-slate-900">{p.label}</div>
                <div className="text-xs text-muted">{p.description}</div>
              </div>
              {connected ? (
                <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-50 px-3 py-1 text-sm font-medium text-emerald-700">
                  ✓ Connected
                </span>
              ) : (
                <button
                  type="button"
                  onClick={() => onConnect(p.id)}
                  disabled={pending === p.id}
                  className="rounded-lg border border-border bg-surface px-3 py-1.5 text-sm font-medium hover:bg-bg disabled:opacity-60"
                >
                  {pending === p.id ? "Connecting…" : `Connect ${p.label}`}
                </button>
              )}
            </div>
          );
        })}
      </div>

      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-2 text-sm text-red-700">
          {error}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Step 5 — Review
// ---------------------------------------------------------------------------

function StepReview() {
  const { watch } = useFormContext<FormValues>();
  const v = watch();
  return (
    <div className="space-y-5">
      <h2 className="text-lg font-semibold text-slate-900">Looks good?</h2>
      <p className="text-sm text-muted">
        Review your info, then submit. You can update availability later from
        Settings.
      </p>

      <div className="grid gap-4 md:grid-cols-2">
        <SummaryCard title="Basics">
          <Row label="Name" value={v.basic.name} />
          <Row label="Email" value={v.basic.email} />
          <Row label="Timezone" value={v.basic.timezone} />
        </SummaryCard>

        <SummaryCard title="Skills">
          {v.skills.length === 0 ? (
            <span className="text-sm text-muted">None</span>
          ) : (
            <ul className="space-y-1 text-sm">
              {v.skills.map((s) => (
                <li key={s.name} className="flex justify-between">
                  <span className="text-slate-900">{s.name}</span>
                  <span className="text-muted">
                    {proficiencyLabel(s.proficiency)}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </SummaryCard>

        <SummaryCard title="Availability">
          <Row
            label="Hours / week"
            value={String(v.availability.hoursPerWeek)}
          />
          <Row
            label="Preferred"
            value={v.availability.preferredTimes.join(", ") || "—"}
          />
          <Row
            label="Blackouts"
            value={
              v.availability.blackouts.length > 0
                ? v.availability.blackouts
                    .map((b) => `${b.start} → ${b.end}`)
                    .join(", ")
                : "None"
            }
          />
        </SummaryCard>

        <SummaryCard title="Connected accounts">
          <Row label="Google" value={v.connections.google ? "✓" : "—"} />
          <Row label="GitHub" value={v.connections.github ? "✓" : "—"} />
          <Row label="Trello" value={v.connections.trello ? "✓" : "—"} />
        </SummaryCard>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Small UI primitives
// ---------------------------------------------------------------------------

function Field({
  label,
  error,
  children,
}: {
  label: string;
  error?: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <label className="mb-1.5 block text-sm font-medium text-slate-900">
        {label}
      </label>
      {children}
      {error && <p className="mt-1 text-xs text-red-600">{error}</p>}
    </div>
  );
}

function SummaryCard({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-lg border border-border bg-bg p-4">
      <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted">
        {title}
      </div>
      <div className="space-y-1">{children}</div>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between text-sm">
      <span className="text-muted">{label}</span>
      <span className="text-right font-medium text-slate-900">{value || "—"}</span>
    </div>
  );
}
