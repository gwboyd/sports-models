"use client";

import { useEffect, type ReactNode } from "react";

export function OverlayPanel({ title, description, onClose, children }: {
  title: string;
  description?: string;
  onClose: () => void;
  children: ReactNode;
}) {
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKeyDown);
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      document.body.style.overflow = previousOverflow;
    };
  }, [onClose]);

  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center bg-slate-950/35 p-0 sm:items-center sm:p-6" role="presentation" onMouseDown={(event) => {
      if (event.target === event.currentTarget) onClose();
    }}>
      <section role="dialog" aria-modal="true" aria-labelledby="overlay-title" className="max-h-[88dvh] w-full overflow-hidden rounded-t-2xl bg-white shadow-2xl sm:max-w-xl sm:rounded-2xl">
        <header className="flex items-start justify-between gap-4 border-b border-[var(--border)] px-5 py-4">
          <div>
            <h2 id="overlay-title" className="text-lg font-semibold text-[var(--ink)]">{title}</h2>
            {description ? <p className="mt-1 text-sm text-[var(--muted)]">{description}</p> : null}
          </div>
          <button type="button" aria-label="Close" onClick={onClose} className="flex min-h-11 min-w-11 items-center justify-center rounded-xl text-2xl text-slate-500 hover:bg-slate-100">×</button>
        </header>
        <div className="max-h-[calc(88dvh-76px)] overflow-y-auto p-5">{children}</div>
      </section>
    </div>
  );
}
