"use client";

import { useRouter } from "next/navigation";
import { useSession } from "@/lib/session-context";

export default function LandingPage() {
  const router = useRouter();
  const { role, projectId } = useSession();

  const onStart = () => {
    if (role === "student") {
      router.push("/student/onboarding");
    } else {
      router.push("/professor/dashboard");
    }
  };

  return (
    <div className="mx-auto max-w-2xl py-16 text-center">
      <h1 className="text-4xl font-semibold tracking-tight text-slate-900 sm:text-5xl">
        Group projects,{" "}
        <span className="text-accent">orchestrated.</span>
      </h1>
      <p className="mt-6 text-lg text-muted">
        SyncUp decomposes your project, assigns tasks fairly, schedules meetings,
        and keeps everyone on track — all on its own.
      </p>

      <div className="mt-10 flex justify-center gap-3">
        <button
          type="button"
          onClick={onStart}
          className="rounded-lg bg-accent px-6 py-3 text-sm font-medium text-white shadow-sm transition hover:bg-accent-hover"
        >
          Get Started
        </button>
      </div>

      <div className="mt-12 rounded-xl border border-border bg-surface p-6 text-left text-sm text-muted shadow-sm">
        <div className="font-medium text-slate-900">Demo session</div>
        <div className="mt-2">
          You are signed in as{" "}
          <span className="font-medium text-slate-900">{role}</span> on project{" "}
          <span className="font-mono text-slate-900">{projectId}</span>. Use the
          role switcher in the top bar to flip between student and professor
          views.
        </div>
      </div>
    </div>
  );
}
