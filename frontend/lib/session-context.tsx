"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

export type Role = "student" | "professor";

interface SessionState {
  role: Role;
  userId: string;
  projectId: string;
  setRole: (r: Role) => void;
  setUserId: (id: string) => void;
  setProjectId: (id: string) => void;
}

const DEFAULT_STATE: Omit<SessionState, "setRole" | "setUserId" | "setProjectId"> = {
  role: "student",
  userId: "stu_demo",
  projectId: "proj-demo",
};

const STORAGE_KEYS = {
  role: "syncup.role",
  userId: "syncup.userId",
  projectId: "syncup.projectId",
} as const;

const SessionContext = createContext<SessionState | null>(null);

function readStorage<T extends string>(key: string, fallback: T): T {
  if (typeof window === "undefined") return fallback;
  const v = window.localStorage.getItem(key);
  return v ? (v as T) : fallback;
}

export function SessionProvider({ children }: { children: ReactNode }) {
  const [role, setRoleState] = useState<Role>(DEFAULT_STATE.role);
  const [userId, setUserIdState] = useState<string>(DEFAULT_STATE.userId);
  const [projectId, setProjectIdState] = useState<string>(DEFAULT_STATE.projectId);

  // Hydrate from localStorage on mount.
  useEffect(() => {
    setRoleState(readStorage<Role>(STORAGE_KEYS.role, DEFAULT_STATE.role));
    setUserIdState(readStorage(STORAGE_KEYS.userId, DEFAULT_STATE.userId));
    setProjectIdState(readStorage(STORAGE_KEYS.projectId, DEFAULT_STATE.projectId));
  }, []);

  const setRole = useCallback((r: Role) => {
    setRoleState(r);
    if (typeof window !== "undefined") window.localStorage.setItem(STORAGE_KEYS.role, r);
  }, []);
  const setUserId = useCallback((id: string) => {
    setUserIdState(id);
    if (typeof window !== "undefined") window.localStorage.setItem(STORAGE_KEYS.userId, id);
  }, []);
  const setProjectId = useCallback((id: string) => {
    setProjectIdState(id);
    if (typeof window !== "undefined")
      window.localStorage.setItem(STORAGE_KEYS.projectId, id);
  }, []);

  const value = useMemo<SessionState>(
    () => ({ role, userId, projectId, setRole, setUserId, setProjectId }),
    [role, userId, projectId, setRole, setUserId, setProjectId],
  );

  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>;
}

export function useSession(): SessionState {
  const ctx = useContext(SessionContext);
  if (ctx === null) {
    // Return defaults if used outside provider — avoids hard crashes during
    // refactors. Page mounts inside <SessionProvider> in app/layout.tsx.
    const noop = (): void => {};
    return {
      ...DEFAULT_STATE,
      setRole: noop,
      setUserId: noop,
      setProjectId: noop,
    };
  }
  return ctx;
}
