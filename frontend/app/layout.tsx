import type { Metadata } from "next";
import { SessionProvider } from "@/lib/session-context";
import { AppShell } from "./AppShell";
import "./globals.css";

export const metadata: Metadata = {
  title: "SyncUp",
  description: "Multi-agent AI project manager for student group work",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="antialiased">
        <SessionProvider>
          <AppShell>{children}</AppShell>
        </SessionProvider>
      </body>
    </html>
  );
}
