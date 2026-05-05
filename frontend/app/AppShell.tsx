"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState, type ReactNode } from "react";
import { useSession, type Role } from "@/lib/session-context";

interface NavLink {
  href: string;
  label: string;
  roles: Role[];
}

const NAV_LINKS: NavLink[] = [
  { href: "/", label: "Home", roles: ["student", "professor"] },
  { href: "/student/onboarding", label: "Onboarding", roles: ["student"] },
  { href: "/student/dashboard", label: "My Dashboard", roles: ["student"] },
  { href: "/student/settings", label: "Settings", roles: ["student"] },
  { href: "/professor/dashboard", label: "Overview", roles: ["professor"] },
];

function initials(id: string): string {
  const slug = id.replace(/^(stu_|prof_)/, "");
  return slug.slice(0, 2).toUpperCase();
}

export function AppShell({ children }: { children: ReactNode }) {
  const { role, userId, setRole } = useSession();
  const pathname = usePathname();
  const [mobileOpen, setMobileOpen] = useState(false);

  const visibleLinks = NAV_LINKS.filter((l) => l.roles.includes(role));

  return (
    <div className="flex min-h-screen">
      <aside
        className={`fixed inset-y-0 left-0 z-30 w-60 transform bg-sidebar text-sidebarText transition-transform md:static md:translate-x-0 ${
          mobileOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <div className="flex h-16 items-center px-6 text-xl font-semibold tracking-tight text-white">
          SyncUp
        </div>
        <nav className="px-3 py-4">
          <ul className="space-y-1">
            {visibleLinks.map((link) => {
              const active =
                pathname === link.href ||
                (link.href !== "/" && pathname.startsWith(link.href));
              return (
                <li key={link.href}>
                  <Link
                    href={link.href}
                    onClick={() => setMobileOpen(false)}
                    className={`block rounded-lg px-3 py-2 text-sm transition ${
                      active
                        ? "bg-sidebarActive text-white"
                        : "text-sidebarText hover:bg-sidebarActive hover:text-white"
                    }`}
                  >
                    {link.label}
                  </Link>
                </li>
              );
            })}
          </ul>
        </nav>
        <div className="absolute bottom-0 left-0 right-0 border-t border-slate-800 px-6 py-4 text-xs text-slate-500">
          v0.1 · demo
        </div>
      </aside>

      {mobileOpen && (
        <div
          className="fixed inset-0 z-20 bg-black/40 md:hidden"
          onClick={() => setMobileOpen(false)}
        />
      )}

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-10 flex h-16 items-center gap-4 border-b border-border bg-surface px-6">
          <button
            type="button"
            aria-label="Toggle navigation"
            className="rounded p-2 text-muted hover:bg-bg md:hidden"
            onClick={() => setMobileOpen((v) => !v)}
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="3" y1="6" x2="21" y2="6" />
              <line x1="3" y1="12" x2="21" y2="12" />
              <line x1="3" y1="18" x2="21" y2="18" />
            </svg>
          </button>

          <div className="ml-auto flex items-center gap-4">
            <RoleSwitcher role={role} onChange={setRole} />
            <div className="flex items-center gap-2">
              <div className="flex h-9 w-9 items-center justify-center rounded-full bg-accent text-sm font-medium text-white">
                {initials(userId)}
              </div>
              <span className="hidden text-sm text-muted sm:inline">
                {userId}
              </span>
            </div>
          </div>
        </header>

        <main className="min-w-0 flex-1 px-6 py-8">{children}</main>
      </div>
    </div>
  );
}

function RoleSwitcher({
  role,
  onChange,
}: {
  role: Role;
  onChange: (r: Role) => void;
}) {
  const base =
    "px-3 py-1.5 text-sm font-medium transition rounded-md";
  return (
    <div className="inline-flex rounded-lg border border-border bg-bg p-0.5">
      <button
        type="button"
        className={`${base} ${
          role === "student"
            ? "bg-surface text-slate-900 shadow-sm"
            : "text-muted hover:text-slate-900"
        }`}
        onClick={() => onChange("student")}
      >
        Student
      </button>
      <button
        type="button"
        className={`${base} ${
          role === "professor"
            ? "bg-surface text-slate-900 shadow-sm"
            : "text-muted hover:text-slate-900"
        }`}
        onClick={() => onChange("professor")}
      >
        Professor
      </button>
    </div>
  );
}
